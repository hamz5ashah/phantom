// Package runner starts and stops llama.cpp's llama-server, one process per
// loaded model, and unloads idle models.
package runner

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"time"
)

// Model describes what to load.
type Model struct {
	Name      string
	Weights   string // GGUF file
	Projector string // optional mmproj file for vision models
	Ctx       int    // context length (0 = default)
	Embedding bool   // serve embeddings instead of chat
	Pooling   string // embedding pooling override ("mean"); empty = the model's own
}

func (m Model) key() string {
	if m.Embedding {
		return m.Name + "#embed"
	}
	return m.Name
}

// Loaded is a running llama-server for one model.
type Loaded struct {
	Model     Model
	URL       string // http://127.0.0.1:<port>
	ExpiresAt time.Time
	cmd       *exec.Cmd
	exited    chan struct{}
	busy      int
}

// Runner manages loaded models.
type Runner struct {
	// Binary is the llama-server executable.
	Binary string
	// MaxLoaded is how many models stay in memory at once (default 1).
	MaxLoaded int
	// DefaultKeepAlive is how long an idle model stays loaded (default 5m).
	DefaultKeepAlive time.Duration
	// ExtraArgs are appended to every llama-server command line.
	ExtraArgs []string
	// Log receives llama-server output (default: discarded).
	Log io.Writer

	mu     sync.Mutex
	loaded map[string]*Loaded
}

// DefaultCtx is the context length when a request does not set num_ctx.
const DefaultCtx = 4096

// Acquire returns a running server for the model, loading it if needed.
// Call the returned release func when the request is finished; keepAlive < 0
// keeps the model loaded until the server stops, 0 unloads it right after.
func (r *Runner) Acquire(ctx context.Context, m Model, keepAlive *time.Duration) (*Loaded, func(), error) {
	if m.Ctx <= 0 {
		m.Ctx = DefaultCtx
	}
	r.mu.Lock()
	if r.loaded == nil {
		r.loaded = map[string]*Loaded{}
	}
	l := r.loaded[m.key()]
	if l != nil && (l.Model != m || l.dead()) {
		r.stopLocked(l)
		l = nil
	}
	if l == nil {
		r.evictLocked()
		var err error
		if l, err = r.start(ctx, m); err != nil {
			r.mu.Unlock()
			return nil, nil, err
		}
		r.loaded[m.key()] = l
	}
	l.busy++
	r.mu.Unlock()

	release := func() {
		r.mu.Lock()
		defer r.mu.Unlock()
		l.busy--
		ka := r.keepAlive()
		if keepAlive != nil {
			ka = *keepAlive
		}
		switch {
		case ka < 0:
			l.ExpiresAt = time.Time{}
		case ka == 0 && l.busy == 0:
			r.stopLocked(l)
		default:
			l.ExpiresAt = time.Now().Add(ka)
			time.AfterFunc(ka, r.reap)
		}
	}
	return l, release, nil
}

func (r *Runner) keepAlive() time.Duration {
	if r.DefaultKeepAlive != 0 {
		return r.DefaultKeepAlive
	}
	return 5 * time.Minute
}

// evictLocked makes room for one more model.
func (r *Runner) evictLocked() {
	max := r.MaxLoaded
	if max <= 0 {
		max = 1
	}
	for len(r.loaded) >= max {
		var victim *Loaded
		for _, l := range r.loaded {
			if l.busy == 0 && (victim == nil || l.ExpiresAt.Before(victim.ExpiresAt)) {
				victim = l
			}
		}
		if victim == nil {
			return // all busy; run alongside them
		}
		r.stopLocked(victim)
	}
}

func (r *Runner) reap() {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := time.Now()
	for _, l := range r.loaded {
		if l.busy == 0 && !l.ExpiresAt.IsZero() && now.After(l.ExpiresAt) {
			r.stopLocked(l)
		}
	}
}

// Running lists loaded models.
func (r *Runner) Running() []Loaded {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]Loaded, 0, len(r.loaded))
	for _, l := range r.loaded {
		out = append(out, Loaded{Model: l.Model, URL: l.URL, ExpiresAt: l.ExpiresAt})
	}
	return out
}

// Unload stops a model if it is loaded.
func (r *Runner) Unload(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, l := range r.loaded {
		if l.Model.Name == name {
			r.stopLocked(l)
		}
	}
}

// StopAll stops every llama-server.
func (r *Runner) StopAll() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, l := range r.loaded {
		r.stopLocked(l)
	}
}

func (r *Runner) stopLocked(l *Loaded) {
	if r.loaded[l.Model.key()] == l {
		delete(r.loaded, l.Model.key())
	}
	if l.cmd.Process != nil && !l.dead() {
		_ = l.cmd.Process.Kill()
		select {
		case <-l.exited:
		case <-time.After(5 * time.Second):
		}
	}
}

func (l *Loaded) dead() bool {
	select {
	case <-l.exited:
		return true
	default:
		return false
	}
}

func (r *Runner) start(ctx context.Context, m Model) (*Loaded, error) {
	if r.Binary == "" {
		return nil, errors.New("llama-server was not found; run `phantom setup` first")
	}
	port, err := freePort()
	if err != nil {
		return nil, err
	}
	args := []string{
		"--model", m.Weights,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--ctx-size", strconv.Itoa(m.Ctx),
		// Use the chat template inside the GGUF file (tool calls, thinking).
		"--jinja",
		// Offload as many layers as fit to the GPU; CPU-only builds ignore it.
		"--n-gpu-layers", "999",
	}
	if m.Projector != "" && !m.Embedding {
		args = append(args, "--mmproj", m.Projector)
	}
	if m.Embedding {
		args = append(args, "--embeddings")
		if m.Pooling != "" {
			args = append(args, "--pooling", m.Pooling)
		}
	}
	args = append(args, r.ExtraArgs...)
	cmd := exec.Command(r.Binary, args...)
	hideWindow(cmd)
	logw := r.Log
	if logw == nil {
		logw = io.Discard
	}
	cmd.Stdout, cmd.Stderr = logw, logw
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("could not start llama-server: %w", err)
	}
	l := &Loaded{Model: m, URL: fmt.Sprintf("http://127.0.0.1:%d", port), cmd: cmd, exited: make(chan struct{})}
	go func() { _ = cmd.Wait(); close(l.exited) }()

	if err := waitHealthy(ctx, l); err != nil {
		_ = cmd.Process.Kill()
		<-l.exited
		return nil, err
	}
	return l, nil
}

// waitHealthy polls /health until the model is loaded (llama-server answers
// 503 while loading).
func waitHealthy(ctx context.Context, l *Loaded) error {
	deadline := time.Now().Add(10 * time.Minute)
	client := &http.Client{Timeout: 2 * time.Second}
	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-l.exited:
			return fmt.Errorf("llama-server exited while loading %s (not enough memory, or an unsupported model?)", l.Model.Name)
		case <-time.After(200 * time.Millisecond):
		}
		resp, err := client.Get(l.URL + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
	}
	return fmt.Errorf("timed out loading %s", l.Model.Name)
}

func freePort() (int, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port, nil
}

// FindBinary locates llama-server: PHANTOM_LLAMA_SERVER, beside the phantom
// executable, the runtime folder `phantom setup` installs to, then PATH.
func FindBinary(runtimeDir string) string {
	name := "llama-server"
	if isWindows {
		name += ".exe"
	}
	if p := os.Getenv("PHANTOM_LLAMA_SERVER"); p != "" {
		return p
	}
	candidates := []string{}
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, joinDir(exe, name))
	}
	if runtimeDir != "" {
		candidates = append(candidates, runtimeDir+string(os.PathSeparator)+name)
	}
	for _, c := range candidates {
		if fi, err := os.Stat(c); err == nil && !fi.IsDir() {
			return c
		}
	}
	if p, err := exec.LookPath(name); err == nil {
		return p
	}
	return ""
}
