// Package server is Phantom's HTTP API: Ollama-compatible endpoints under
// /api and OpenAI-compatible endpoints under /v1.
package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/hamz5ashah/phantom/internal/api"
	"github.com/hamz5ashah/phantom/internal/registry"
	"github.com/hamz5ashah/phantom/internal/runner"
	"github.com/hamz5ashah/phantom/internal/store"
)

// Version is set at build time.
var Version = "0.1.0"

// Server holds what the handlers need.
type Server struct {
	Store    *store.Store
	Runner   *runner.Runner
	Registry *registry.Client
	// upstream talks to llama-server; no timeout (generations can be long).
	upstream http.Client
}

// Handler returns the HTTP routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, _ *http.Request) { io.WriteString(w, "Phantom is running") })
	mux.HandleFunc("HEAD /{$}", func(http.ResponseWriter, *http.Request) {})
	mux.HandleFunc("GET /api/version", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, map[string]string{"version": Version}) })
	mux.HandleFunc("GET /api/tags", s.tags)
	mux.HandleFunc("GET /api/ps", s.ps)
	mux.HandleFunc("POST /api/show", s.show)
	mux.HandleFunc("POST /api/pull", s.pull)
	mux.HandleFunc("DELETE /api/delete", s.del)
	mux.HandleFunc("POST /api/chat", s.chat)
	mux.HandleFunc("POST /api/generate", s.generate)
	mux.HandleFunc("POST /api/embed", s.embed)
	mux.HandleFunc("POST /api/embeddings", s.embed)
	mux.HandleFunc("GET /v1/models", s.openAIModels)
	mux.HandleFunc("POST /v1/chat/completions", s.openAIProxy)
	mux.HandleFunc("POST /v1/completions", s.openAIProxy)
	mux.HandleFunc("POST /v1/embeddings", s.openAIProxy)
	return cors(mux)
}

// --- model management ---

type modelInfo struct {
	Name       string        `json:"name"`
	Model      string        `json:"model"`
	ModifiedAt time.Time     `json:"modified_at"`
	Size       int64         `json:"size"`
	Digest     string        `json:"digest"`
	Details    store.Details `json:"details"`
}

func (s *Server) info(m *store.Model) modelInfo {
	digest := ""
	if l, ok := m.Manifest.Layer(store.MediaModel); ok {
		digest = strings.TrimPrefix(l.Digest, "sha256:")
	}
	n := m.Name.String()
	return modelInfo{Name: n, Model: n, ModifiedAt: m.Modified, Size: m.Manifest.Size(), Digest: digest, Details: s.Store.Details(m)}
}

func (s *Server) tags(w http.ResponseWriter, _ *http.Request) {
	models, err := s.Store.List()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	out := []modelInfo{}
	for _, m := range models {
		out = append(out, s.info(m))
	}
	writeJSON(w, map[string]any{"models": out})
}

func (s *Server) ps(w http.ResponseWriter, _ *http.Request) {
	out := []map[string]any{}
	for _, l := range s.Runner.Running() {
		item := map[string]any{"name": l.Model.Name, "model": l.Model.Name, "expires_at": l.ExpiresAt, "context_length": l.Model.Ctx}
		if m, err := s.Store.Get(l.Model.Name); err == nil {
			i := s.info(m)
			item["size"], item["digest"], item["details"] = i.Size, i.Digest, i.Details
		}
		out = append(out, item)
	}
	writeJSON(w, map[string]any{"models": out})
}

type nameReq struct {
	Model string `json:"model"`
	Name  string `json:"name"` // older clients
}

func (n nameReq) get() string {
	if n.Model != "" {
		return n.Model
	}
	return n.Name
}

func (s *Server) show(w http.ResponseWriter, r *http.Request) {
	var req nameReq
	if !readJSON(w, r, &req) {
		return
	}
	m, err := s.Store.Get(req.get())
	if err != nil {
		writeErr(w, http.StatusNotFound, err)
		return
	}
	i := s.info(m)
	caps := []string{"completion", "tools"}
	if _, ok := s.Store.ProjectorPath(m); ok {
		caps = append(caps, "vision")
	}
	writeJSON(w, map[string]any{
		"modelfile": "", "parameters": "", "template": "",
		"details": i.Details, "model_info": map[string]any{}, "capabilities": caps, "modified_at": i.ModifiedAt,
	})
}

func (s *Server) del(w http.ResponseWriter, r *http.Request) {
	var req nameReq
	if !readJSON(w, r, &req) {
		return
	}
	s.Runner.Unload(mustName(req.get()))
	if err := s.Store.Delete(req.get()); err != nil {
		writeErr(w, http.StatusNotFound, err)
		return
	}
	w.WriteHeader(http.StatusOK)
}

func (s *Server) pull(w http.ResponseWriter, r *http.Request) {
	var req struct {
		nameReq
		Stream *bool `json:"stream"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	stream := req.Stream == nil || *req.Stream
	if !stream {
		if err := s.Registry.Pull(r.Context(), req.get(), nil); err != nil {
			writeErr(w, http.StatusInternalServerError, err)
			return
		}
		writeJSON(w, map[string]string{"status": "success"})
		return
	}
	nd := newNDJSON(w)
	err := s.Registry.Pull(r.Context(), req.get(), func(p registry.Progress) { nd.send(p) })
	if err != nil {
		nd.send(map[string]string{"error": err.Error()})
	}
}

// --- generation ---

func (s *Server) chat(w http.ResponseWriter, r *http.Request) {
	var req api.ChatRequest
	if !readJSON(w, r, &req) {
		return
	}
	s.complete(w, r, &req, false)
}

func (s *Server) generate(w http.ResponseWriter, r *http.Request) {
	var req api.GenerateRequest
	if !readJSON(w, r, &req) {
		return
	}
	// An empty prompt just loads (or with keep_alive 0, unloads) the model.
	if req.Prompt == "" && len(req.Images) == 0 {
		s.loadOnly(w, r, req.Model, req.KeepAlive)
		return
	}
	s.complete(w, r, api.FromGenerate(&req), true)
}

func (s *Server) loadOnly(w http.ResponseWriter, r *http.Request, model string, ka *api.Duration) {
	if ka != nil && ka.Duration == 0 {
		s.Runner.Unload(mustName(model))
		writeJSON(w, api.ChatResponse{Model: model, CreatedAt: api.Now(), Done: true, DoneReason: "unload"})
		return
	}
	_, release, _, err := s.acquire(r.Context(), model, 0, ka, false)
	if err != nil {
		writeErr(w, statusFor(err), err)
		return
	}
	release()
	empty := ""
	writeJSON(w, api.ChatResponse{Model: model, CreatedAt: api.Now(), Response: &empty, Done: true, DoneReason: "load"})
}

func (s *Server) acquire(ctx context.Context, model string, numCtx int, ka *api.Duration, embedding bool, pooling ...string) (*runner.Loaded, func(), time.Duration, error) {
	m, err := s.Store.Get(model)
	if err != nil {
		return nil, nil, 0, err
	}
	weights, err := s.Store.WeightsPath(m)
	if err != nil {
		return nil, nil, 0, err
	}
	proj, _ := s.Store.ProjectorPath(m)
	var keep *time.Duration
	if ka != nil {
		keep = &ka.Duration
	}
	rm := runner.Model{Name: m.Name.String(), Weights: weights, Projector: proj, Ctx: numCtx, Embedding: embedding}
	if len(pooling) > 0 {
		rm.Pooling = pooling[0]
	}
	start := time.Now()
	l, release, err := s.Runner.Acquire(ctx, rm, keep)
	return l, release, time.Since(start), err
}

func (s *Server) complete(w http.ResponseWriter, r *http.Request, req *api.ChatRequest, generate bool) {
	start := time.Now()
	oa, err := api.ToOpenAI(req)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	l, release, loadDur, err := s.acquire(r.Context(), req.Model, req.Options.NumCtx, req.KeepAlive, false)
	if err != nil {
		writeErr(w, statusFor(err), err)
		return
	}
	defer release()

	body, _ := json.Marshal(oa)
	up, err := http.NewRequestWithContext(r.Context(), http.MethodPost, l.URL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		writeErr(w, http.StatusInternalServerError, err)
		return
	}
	up.Header.Set("Content-Type", "application/json")
	resp, err := s.upstream.Do(up)
	if err != nil {
		writeErr(w, http.StatusBadGateway, fmt.Errorf("llama-server: %w", err))
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		writeErr(w, resp.StatusCode, errors.New(upstreamError(msg)))
		return
	}

	asm := &api.Assembler{Model: req.Model, Generate: generate, Start: start, LoadDur: loadDur}
	if !oa.Stream {
		var c api.OAResponse
		if err := json.NewDecoder(resp.Body).Decode(&c); err != nil {
			writeErr(w, http.StatusBadGateway, err)
			return
		}
		writeJSON(w, asm.Whole(&c))
		return
	}
	nd := newNDJSON(w)
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64*1024), 8*1024*1024)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok {
			continue
		}
		if data == "[DONE]" {
			break
		}
		var c api.OAResponse
		if json.Unmarshal([]byte(data), &c) != nil {
			continue
		}
		if line := asm.Chunk(&c); line != nil {
			nd.send(line)
		}
	}
	if err := sc.Err(); err != nil && r.Context().Err() == nil {
		nd.send(map[string]string{"error": err.Error()})
		return
	}
	nd.send(asm.Final())
}

// --- embeddings ---

// embed serves both Ollama forms: /api/embed {"input": string|[]string} ->
// {"embeddings": [[...]]} and the older /api/embeddings {"prompt": string} ->
// {"embedding": [...]}.
func (s *Server) embed(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model     string          `json:"model"`
		Input     json.RawMessage `json:"input"`
		Prompt    string          `json:"prompt"`
		KeepAlive *api.Duration   `json:"keep_alive"`
		Options   api.Options     `json:"options"`
	}
	if !readJSON(w, r, &req) {
		return
	}
	legacy := r.URL.Path == "/api/embeddings"
	var inputs []string
	if legacy {
		inputs = []string{req.Prompt}
	} else if len(req.Input) > 0 {
		var one string
		if json.Unmarshal(req.Input, &one) == nil {
			inputs = []string{one}
		} else if err := json.Unmarshal(req.Input, &inputs); err != nil {
			writeErr(w, http.StatusBadRequest, errors.New(`"input" must be a string or a list of strings`))
			return
		}
	}
	if len(inputs) == 0 {
		writeJSON(w, map[string]any{"model": req.Model, "embeddings": [][]float64{}})
		return
	}
	start := time.Now()
	body, _ := json.Marshal(map[string]any{"input": inputs})
	var (
		resp    *http.Response
		loadDur time.Duration
	)
	// Chat models have no pooling of their own, which llama.cpp refuses for
	// OpenAI-style embeddings; reload such a model with mean pooling once.
	for _, pooling := range []string{"", "mean"} {
		l, release, ld, err := s.acquire(r.Context(), req.Model, req.Options.NumCtx, req.KeepAlive, true, pooling)
		if err != nil {
			writeErr(w, statusFor(err), err)
			return
		}
		var once sync.Once
		releaseOnce := func() { once.Do(release) }
		defer releaseOnce()
		loadDur += ld
		up, _ := http.NewRequestWithContext(r.Context(), http.MethodPost, l.URL+"/v1/embeddings", bytes.NewReader(body))
		up.Header.Set("Content-Type", "application/json")
		resp, err = s.upstream.Do(up)
		if err != nil {
			writeErr(w, http.StatusBadGateway, fmt.Errorf("llama-server: %w", err))
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			break
		}
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		text := upstreamError(msg)
		if pooling == "" && strings.Contains(strings.ToLower(text), "pooling") {
			releaseOnce()
			s.Runner.Unload(mustName(req.Model))
			continue
		}
		writeErr(w, resp.StatusCode, errors.New(text))
		return
	}
	var out struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		writeErr(w, http.StatusBadGateway, err)
		return
	}
	vecs := make([][]float64, len(inputs))
	for _, d := range out.Data {
		if d.Index >= 0 && d.Index < len(vecs) {
			vecs[d.Index] = d.Embedding
		}
	}
	if legacy {
		writeJSON(w, map[string]any{"embedding": vecs[0]})
		return
	}
	writeJSON(w, map[string]any{
		"model": req.Model, "embeddings": vecs, "prompt_eval_count": out.Usage.PromptTokens,
		"total_duration": int64(time.Since(start)), "load_duration": int64(loadDur),
	})
}

// --- OpenAI-compatible passthrough ---

func (s *Server) openAIModels(w http.ResponseWriter, _ *http.Request) {
	models, _ := s.Store.List()
	data := []map[string]any{}
	for _, m := range models {
		data = append(data, map[string]any{"id": m.Name.String(), "object": "model", "created": m.Modified.Unix(), "owned_by": "phantom"})
	}
	writeJSON(w, map[string]any{"object": "list", "data": data})
}

func (s *Server) openAIProxy(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 64<<20))
	if err != nil {
		writeErr(w, http.StatusBadRequest, err)
		return
	}
	var head struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &head); err != nil || head.Model == "" {
		writeErr(w, http.StatusBadRequest, errors.New(`request needs a "model"`))
		return
	}
	l, release, _, err := s.acquire(r.Context(), head.Model, 0, nil, r.URL.Path == "/v1/embeddings")
	if err != nil {
		writeErr(w, statusFor(err), err)
		return
	}
	defer release()
	up, _ := http.NewRequestWithContext(r.Context(), http.MethodPost, l.URL+r.URL.Path, bytes.NewReader(body))
	up.Header.Set("Content-Type", "application/json")
	resp, err := s.upstream.Do(up)
	if err != nil {
		writeErr(w, http.StatusBadGateway, err)
		return
	}
	defer resp.Body.Close()
	for _, k := range []string{"Content-Type", "Cache-Control"} {
		if v := resp.Header.Get(k); v != "" {
			w.Header().Set(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 16*1024)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			return
		}
	}
}

// --- helpers ---

func mustName(model string) string {
	if n, err := store.ParseName(model); err == nil {
		return n.String()
	}
	return model
}

func statusFor(err error) int {
	if strings.Contains(err.Error(), "not found") {
		return http.StatusNotFound
	}
	return http.StatusInternalServerError
}

func upstreamError(body []byte) string {
	var e struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &e) == nil && e.Error.Message != "" {
		return e.Error.Message
	}
	return strings.TrimSpace(string(body))
}

func readJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(io.LimitReader(r.Body, 64<<20)).Decode(v); err != nil {
		writeErr(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, code int, err error) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
}

type ndjson struct {
	w http.ResponseWriter
	f http.Flusher
}

func newNDJSON(w http.ResponseWriter) *ndjson {
	w.Header().Set("Content-Type", "application/x-ndjson")
	f, _ := w.(http.Flusher)
	return &ndjson{w: w, f: f}
}

func (n *ndjson) send(v any) {
	b, _ := json.Marshal(v)
	n.w.Write(append(b, '\n'))
	if n.f != nil {
		n.f.Flush()
	}
}

// cors lets browser apps on this machine call the API.
func cors(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" && isLocalOrigin(origin) {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		h.ServeHTTP(w, r)
	})
}

func isLocalOrigin(o string) bool {
	for _, p := range []string{"http://localhost", "http://127.0.0.1", "https://localhost", "tauri://", "http://tauri.localhost", "https://tauri.localhost", "app://"} {
		if strings.HasPrefix(o, p) {
			return true
		}
	}
	return false
}
