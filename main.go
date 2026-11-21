// Phantom runs open AI models on your own computer, with an Ollama-compatible API.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"text/tabwriter"
	"time"

	"github.com/hamz5ashah/phantom/internal/api"
	"github.com/hamz5ashah/phantom/internal/registry"
	"github.com/hamz5ashah/phantom/internal/runner"
	"github.com/hamz5ashah/phantom/internal/server"
	"github.com/hamz5ashah/phantom/internal/setup"
	"github.com/hamz5ashah/phantom/internal/store"
)

const usage = `Phantom - run AI models on your own computer.

Usage:
  phantom serve                 Start the API server (Ollama-compatible)
  phantom run MODEL [PROMPT]    Chat with a model (pulls it if needed)
  phantom pull MODEL            Download a model
  phantom list                  List downloaded models
  phantom ps                    List loaded models
  phantom rm MODEL              Delete a model
  phantom setup [--backend B]   Download llama.cpp (B: vulkan, cpu, cuda, metal)
  phantom version               Show the version

Models: any Ollama library name (qwen3, llama3.2:3b, gemma3:4b ...) or a
Hugging Face GGUF repo (hf.co/owner/repo:Q4_K_M).

Environment:
  PHANTOM_HOST          Address to listen on / connect to (default 127.0.0.1:11435)
  PHANTOM_MODELS        Models folder (default ~/.phantom/models)
  PHANTOM_KEEP_ALIVE    How long an idle model stays loaded (default 5m)
  PHANTOM_MAX_LOADED    Models kept in memory at once (default 1)
  PHANTOM_LLAMA_SERVER  Use this llama-server instead of the installed one
`

func main() {
	if len(os.Args) < 2 {
		fmt.Print(usage)
		os.Exit(2)
	}
	cmd, args := os.Args[1], os.Args[2:]
	var err error
	switch cmd {
	case "serve", "start":
		err = serve()
	case "run":
		err = run(args)
	case "pull":
		err = pull(args)
	case "list", "ls":
		err = list()
	case "ps":
		err = ps()
	case "rm", "delete":
		err = rm(args)
	case "setup":
		err = doSetup(args)
	case "version", "--version", "-v":
		fmt.Println("phantom version", server.Version)
	case "help", "--help", "-h":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n%s", cmd, usage)
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}

func host() string {
	h := os.Getenv("PHANTOM_HOST")
	if h == "" {
		h = "127.0.0.1:11435"
	}
	return strings.TrimPrefix(strings.TrimPrefix(h, "http://"), "https://")
}

func baseURL() string { return "http://" + host() }

// --- server ---

func newServer() (*server.Server, error) {
	st, err := store.Default()
	if err != nil {
		return nil, err
	}
	bin := runner.FindBinary(setup.RuntimeDir())
	if bin == "" {
		fmt.Fprintln(os.Stderr, "llama.cpp is not installed yet; downloading it (one time)...")
		bin, err = setup.Install(context.Background(), setup.RuntimeDir(), "", func(s string) { fmt.Fprintln(os.Stderr, s) })
		if err != nil {
			return nil, fmt.Errorf("could not install llama.cpp: %w (run `phantom setup` to retry)", err)
		}
	}
	r := &runner.Runner{Binary: bin}
	if v := os.Getenv("PHANTOM_KEEP_ALIVE"); v != "" {
		var d api.Duration
		if err := json.Unmarshal([]byte(fmt.Sprintf("%q", v)), &d); err == nil {
			r.DefaultKeepAlive = d.Duration
		}
	}
	if v := os.Getenv("PHANTOM_MAX_LOADED"); v != "" {
		fmt.Sscan(v, &r.MaxLoaded)
	}
	if os.Getenv("PHANTOM_DEBUG") != "" {
		r.Log = os.Stderr
	}
	return &server.Server{Store: st, Runner: r, Registry: &registry.Client{Store: st}}, nil
}

func serve() error {
	srv, err := newServer()
	if err != nil {
		return err
	}
	ln, err := net.Listen("tcp", host())
	if err != nil {
		return fmt.Errorf("could not listen on %s: %w", host(), err)
	}
	fmt.Fprintf(os.Stderr, "Phantom %s listening on http://%s\n", server.Version, host())
	hs := &http.Server{Handler: srv.Handler()}
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-stop
		srv.Runner.StopAll()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = hs.Shutdown(ctx)
	}()
	err = hs.Serve(ln)
	srv.Runner.StopAll()
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}

// ensureServer starts the server inside this process when none is running.
func ensureServer() error {
	if reachable() {
		return nil
	}
	errc := make(chan error, 1)
	go func() { errc <- serve() }()
	for i := 0; i < 600; i++ {
		select {
		case err := <-errc:
			return err
		case <-time.After(100 * time.Millisecond):
		}
		if reachable() {
			return nil
		}
	}
	return errors.New("the Phantom server did not start")
}

func reachable() bool {
	c := http.Client{Timeout: 500 * time.Millisecond}
	resp, err := c.Get(baseURL() + "/api/version")
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// --- client commands ---

func post(path string, body any) (*http.Response, error) {
	b, _ := json.Marshal(body)
	resp, err := http.Post(baseURL()+path, "application/json", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		var e struct{ Error string }
		raw, _ := io.ReadAll(resp.Body)
		if json.Unmarshal(raw, &e) == nil && e.Error != "" {
			return nil, errors.New(e.Error)
		}
		return nil, fmt.Errorf("%s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}
	return resp, nil
}

func pull(args []string) error {
	if len(args) != 1 {
		return errors.New("usage: phantom pull MODEL")
	}
	if err := ensureServer(); err != nil {
		return err
	}
	return pullModel(args[0])
}

func pullModel(name string) error {
	resp, err := post("/api/pull", map[string]any{"model": name})
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	dec := json.NewDecoder(resp.Body)
	last := ""
	for {
		var p struct {
			registry.Progress
			Error string `json:"error"`
		}
		if err := dec.Decode(&p); err == io.EOF {
			break
		} else if err != nil {
			return err
		}
		if p.Error != "" {
			fmt.Fprintln(os.Stderr)
			return errors.New(p.Error)
		}
		if p.Total > 0 {
			fmt.Fprintf(os.Stderr, "\r%s  %3d%%  %s / %s   ", p.Status, p.Completed*100/p.Total, mb(p.Completed), mb(p.Total))
			last = p.Status
			continue
		}
		if last != "" {
			fmt.Fprintln(os.Stderr)
			last = ""
		}
		fmt.Fprintln(os.Stderr, p.Status)
	}
	return nil
}

func mb(n int64) string {
	if n >= 1<<30 {
		return fmt.Sprintf("%.1f GB", float64(n)/(1<<30))
	}
	return fmt.Sprintf("%d MB", n>>20)
}

func list() error {
	if err := ensureServer(); err != nil {
		return err
	}
	resp, err := http.Get(baseURL() + "/api/tags")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	var out struct {
		Models []struct {
			Name       string    `json:"name"`
			Size       int64     `json:"size"`
			Digest     string    `json:"digest"`
			ModifiedAt time.Time `json:"modified_at"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return err
	}
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(tw, "NAME\tID\tSIZE\tMODIFIED")
	for _, m := range out.Models {
		id := m.Digest
		if len(id) > 12 {
			id = id[:12]
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", m.Name, id, mb(m.Size), m.ModifiedAt.Format("2006-01-02 15:04"))
	}
	return tw.Flush()
}

func ps() error {
	if !reachable() {
		fmt.Println("No models loaded (the server is not running).")
		return nil
	}
	resp, err := http.Get(baseURL() + "/api/ps")
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	var out struct {
		Models []struct {
			Name      string    `json:"name"`
			ExpiresAt time.Time `json:"expires_at"`
			Ctx       int       `json:"context_length"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return err
	}
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(tw, "NAME\tCONTEXT\tUNTIL")
	for _, m := range out.Models {
		until := "forever"
		if !m.ExpiresAt.IsZero() {
			until = time.Until(m.ExpiresAt).Round(time.Second).String()
		}
		fmt.Fprintf(tw, "%s\t%d\t%s\n", m.Name, m.Ctx, until)
	}
	return tw.Flush()
}

func rm(args []string) error {
	if len(args) != 1 {
		return errors.New("usage: phantom rm MODEL")
	}
	if err := ensureServer(); err != nil {
		return err
	}
	b, _ := json.Marshal(map[string]string{"model": args[0]})
	req, _ := http.NewRequest(http.MethodDelete, baseURL()+"/api/delete", bytes.NewReader(b))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		var e struct{ Error string }
		_ = json.NewDecoder(resp.Body).Decode(&e)
		return errors.New(e.Error)
	}
	fmt.Println("deleted", args[0])
	return nil
}

func doSetup(args []string) error {
	backend := ""
	for i := 0; i < len(args); i++ {
		if args[i] == "--backend" && i+1 < len(args) {
			backend = args[i+1]
			i++
		} else if v, ok := strings.CutPrefix(args[i], "--backend="); ok {
			backend = v
		}
	}
	_, err := setup.Install(context.Background(), setup.RuntimeDir(), backend, func(s string) { fmt.Println(s) })
	return err
}

func run(args []string) error {
	if len(args) < 1 {
		return errors.New("usage: phantom run MODEL [PROMPT]")
	}
	model := args[0]
	if err := ensureServer(); err != nil {
		return err
	}
	resp, err := post("/api/show", map[string]string{"model": model})
	if err != nil {
		if !strings.Contains(err.Error(), "not found") {
			return err
		}
		if err := pullModel(model); err != nil {
			return err
		}
	} else {
		resp.Body.Close()
	}

	history := []api.Message{}
	ask := func(prompt string) error {
		history = append(history, api.Message{Role: "user", Content: prompt})
		resp, err := post("/api/chat", map[string]any{"model": model, "messages": history})
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		dec := json.NewDecoder(resp.Body)
		var reply strings.Builder
		thinking := false
		for {
			var line struct {
				api.ChatResponse
				Error string `json:"error"`
			}
			if err := dec.Decode(&line); err == io.EOF {
				break
			} else if err != nil {
				return err
			}
			if line.Error != "" {
				return errors.New(line.Error)
			}
			if line.Message == nil {
				continue
			}
			if t := line.Message.Thinking; t != "" {
				if !thinking {
					fmt.Print("\x1b[2mThinking...\n")
					thinking = true
				}
				fmt.Print(t)
			}
			if c := line.Message.Content; c != "" {
				if thinking {
					fmt.Print("\x1b[0m\n\n")
					thinking = false
				}
				fmt.Print(c)
				reply.WriteString(c)
			}
		}
		if thinking {
			fmt.Print("\x1b[0m")
		}
		fmt.Println()
		history = append(history, api.Message{Role: "assistant", Content: reply.String()})
		return nil
	}

	if len(args) > 1 {
		return ask(strings.Join(args[1:], " "))
	}
	fmt.Fprintln(os.Stderr, "Type a message (/bye to quit, /clear to reset the chat).")
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 64*1024), 1<<20)
	for {
		fmt.Print(">>> ")
		if !in.Scan() {
			fmt.Println()
			return nil
		}
		text := strings.TrimSpace(in.Text())
		switch text {
		case "":
			continue
		case "/bye", "/exit", "/quit":
			return nil
		case "/clear":
			history = nil
			fmt.Println("Cleared.")
			continue
		}
		if err := ask(text); err != nil {
			fmt.Fprintln(os.Stderr, "Error:", err)
			history = history[:len(history)-1]
		}
	}
}
