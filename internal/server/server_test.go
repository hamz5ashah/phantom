package server

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/hamz5ashah/phantom/internal/registry"
	"github.com/hamz5ashah/phantom/internal/runner"
	"github.com/hamz5ashah/phantom/internal/store"
)

var fakeLlama string

func TestMain(m *testing.M) {
	dir, _ := os.MkdirTemp("", "fakellama")
	fakeLlama = filepath.Join(dir, "llama-server")
	if runtime.GOOS == "windows" {
		fakeLlama += ".exe"
	}
	if out, err := exec.Command("go", "build", "-o", fakeLlama, "./testdata/fakellama").CombinedOutput(); err != nil {
		fmt.Println(string(out))
		os.Exit(1)
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

func digest(b []byte) string {
	h := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(h[:])
}

// fakeRegistry serves one model the way registry.ollama.ai does.
func fakeRegistry(t *testing.T) *httptest.Server {
	weights := []byte("GGUF fake weights for tests")
	config := []byte(`{"model_family":"qwen3","model_type":"0.6B","file_type":"Q4_K_M"}`)
	blobs := map[string][]byte{digest(weights): weights, digest(config): config}
	manifest := store.Manifest{
		SchemaVersion: 2,
		Config:        store.Layer{MediaType: store.MediaConfig, Digest: digest(config), Size: int64(len(config))},
		Layers: []store.Layer{
			{MediaType: store.MediaModel, Digest: digest(weights), Size: int64(len(weights))},
			{MediaType: "application/vnd.ollama.image.template", Digest: digest([]byte("tmpl")), Size: 4},
		},
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v2/library/qwen3/manifests/0.6b", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(manifest)
	})
	mux.HandleFunc("GET /v2/library/qwen3/blobs/{digest}", func(w http.ResponseWriter, r *http.Request) {
		b, ok := blobs[r.PathValue("digest")]
		if !ok {
			http.NotFound(w, r)
			return
		}
		http.ServeContent(w, r, "", time.Time{}, bytes.NewReader(b))
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func newTestServer(t *testing.T) (*httptest.Server, *Server) {
	st := &store.Store{Dir: t.TempDir()}
	os.MkdirAll(filepath.Join(st.Dir, "blobs"), 0o755)
	reg := fakeRegistry(t)
	s := &Server{
		Store:    st,
		Runner:   &runner.Runner{Binary: fakeLlama},
		Registry: &registry.Client{Store: st, BaseURL: reg.URL},
	}
	hs := httptest.NewServer(s.Handler())
	t.Cleanup(func() { hs.Close(); s.Runner.StopAll() })
	return hs, s
}

func postJSON(t *testing.T, url string, body any) *http.Response {
	t.Helper()
	b, _ := json.Marshal(body)
	resp, err := http.Post(url, "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func lines(t *testing.T, r io.Reader) []map[string]any {
	t.Helper()
	var out []map[string]any
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		var m map[string]any
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			t.Fatalf("bad ndjson line %q: %v", sc.Text(), err)
		}
		out = append(out, m)
	}
	return out
}

func TestPullChatGenerateAndManage(t *testing.T) {
	hs, s := newTestServer(t)

	// Pull streams progress and ends with success.
	resp := postJSON(t, hs.URL+"/api/pull", map[string]any{"model": "qwen3:0.6b"})
	got := lines(t, resp.Body)
	resp.Body.Close()
	if last := got[len(got)-1]; last["status"] != "success" {
		t.Fatalf("pull did not succeed: %v", got)
	}

	// Tags lists it with details from the config blob.
	resp, _ = http.Get(hs.URL + "/api/tags")
	var tags struct {
		Models []struct {
			Name    string        `json:"name"`
			Details store.Details `json:"details"`
		} `json:"models"`
	}
	json.NewDecoder(resp.Body).Decode(&tags)
	resp.Body.Close()
	if len(tags.Models) != 1 || tags.Models[0].Name != "qwen3:0.6b" || tags.Models[0].Details.QuantizationLevel != "Q4_K_M" {
		t.Fatalf("unexpected tags: %+v", tags)
	}

	// Streamed chat: thinking, content, then a done line with token counts.
	resp = postJSON(t, hs.URL+"/api/chat", map[string]any{
		"model":    "qwen3:0.6b",
		"messages": []map[string]string{{"role": "user", "content": "hello"}},
	})
	chat := lines(t, resp.Body)
	resp.Body.Close()
	var text, thinking strings.Builder
	for _, l := range chat {
		if m, ok := l["message"].(map[string]any); ok {
			text.WriteString(fmt.Sprint(m["content"]))
			if th, ok := m["thinking"].(string); ok {
				thinking.WriteString(th)
			}
		}
	}
	final := chat[len(chat)-1]
	if text.String() != "echo: hello" || thinking.String() != "hmm" || final["done"] != true || final["eval_count"] != float64(3) {
		t.Fatalf("unexpected chat stream: %v", chat)
	}

	// Streamed tool call fragments are joined into one call with object arguments.
	resp = postJSON(t, hs.URL+"/api/chat", map[string]any{
		"model":    "qwen3:0.6b",
		"messages": []map[string]string{{"role": "user", "content": "weather?"}},
		"tools":    []any{map[string]any{"type": "function", "function": map[string]any{"name": "get_weather"}}},
	})
	chat = lines(t, resp.Body)
	resp.Body.Close()
	calls := chat[len(chat)-1]["message"].(map[string]any)["tool_calls"].([]any)
	fn := calls[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "get_weather" || fn["arguments"].(map[string]any)["city"] != "Lahore" {
		t.Fatalf("unexpected tool call: %v", calls)
	}

	// Non-streamed generate.
	resp = postJSON(t, hs.URL+"/api/generate", map[string]any{"model": "qwen3:0.6b", "prompt": "hi", "stream": false})
	var gen map[string]any
	json.NewDecoder(resp.Body).Decode(&gen)
	resp.Body.Close()
	if gen["response"] != "echo: hi" || gen["done"] != true {
		t.Fatalf("unexpected generate: %v", gen)
	}

	// OpenAI-compatible passthrough.
	resp = postJSON(t, hs.URL+"/v1/chat/completions", map[string]any{
		"model": "qwen3:0.6b", "messages": []map[string]string{{"role": "user", "content": "yo"}},
	})
	var oa map[string]any
	json.NewDecoder(resp.Body).Decode(&oa)
	resp.Body.Close()
	if !strings.Contains(fmt.Sprint(oa["choices"]), "echo: yo") {
		t.Fatalf("unexpected /v1 reply: %v", oa)
	}

	// The model is loaded; keep_alive 0 unloads it.
	if n := len(s.Runner.Running()); n != 1 {
		t.Fatalf("expected 1 loaded model, got %d", n)
	}
	resp = postJSON(t, hs.URL+"/api/generate", map[string]any{"model": "qwen3:0.6b", "keep_alive": 0})
	resp.Body.Close()
	if n := len(s.Runner.Running()); n != 0 {
		t.Fatalf("expected the model to be unloaded, %d still loaded", n)
	}

	// Delete removes the model and its blobs.
	b, _ := json.Marshal(map[string]string{"model": "qwen3:0.6b"})
	req, _ := http.NewRequest(http.MethodDelete, hs.URL+"/api/delete", bytes.NewReader(b))
	resp, _ = http.DefaultClient.Do(req)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("delete failed: %s", resp.Status)
	}
	if models, _ := s.Store.List(); len(models) != 0 {
		t.Fatalf("model still listed after delete")
	}
	if blobs, _ := os.ReadDir(filepath.Join(s.Store.Dir, "blobs")); len(blobs) != 0 {
		t.Fatalf("blobs left after delete: %v", blobs)
	}
}

func TestUnknownModelIs404(t *testing.T) {
	hs, _ := newTestServer(t)
	resp := postJSON(t, hs.URL+"/api/chat", map[string]any{"model": "nope", "messages": []any{}})
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404, got %s", resp.Status)
	}
}
