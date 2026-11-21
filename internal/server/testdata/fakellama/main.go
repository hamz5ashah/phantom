// fakellama imitates llama-server's /health and /v1/chat/completions for tests.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
)

func main() {
	host := flag.String("host", "127.0.0.1", "")
	port := flag.Int("port", 0, "")
	model := flag.String("model", "", "")
	flag.Int("ctx-size", 0, "")
	flag.Bool("jinja", false, "")
	flag.Int("n-gpu-layers", 0, "")
	flag.String("mmproj", "", "")
	embeddings := flag.Bool("embeddings", false, "")
	pooling := flag.String("pooling", "", "")
	flag.Parse()
	if _, err := os.Stat(*model); err != nil {
		fmt.Fprintln(os.Stderr, "model missing:", err)
		os.Exit(1)
	}
	http.HandleFunc("/v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
		if !*embeddings {
			http.Error(w, `{"error":{"message":"embeddings not enabled"}}`, http.StatusNotImplemented)
			return
		}
		// Like llama.cpp with a chat model: no pooling unless one is set.
		if *pooling == "" {
			http.Error(w, `{"error":{"message":"Pooling type 'none' is not OAI compatible. Please use a different pooling type"}}`, http.StatusBadRequest)
			return
		}
		var req struct {
			Input []string `json:"input"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		data := []any{}
		for i, in := range req.Input {
			data = append(data, map[string]any{"index": i, "embedding": []float64{float64(len(in)), 0.5}})
		}
		json.NewEncoder(w).Encode(map[string]any{"data": data, "usage": map[string]int{"prompt_tokens": 4}})
	})
	http.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte(`{"status":"ok"}`)) })
	http.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Stream   bool `json:"stream"`
			Messages []struct {
				Content any `json:"content"`
			} `json:"messages"`
			Tools json.RawMessage `json:"tools"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		last := fmt.Sprint(req.Messages[len(req.Messages)-1].Content)
		reply := "echo: " + last
		if !req.Stream {
			msg := map[string]any{"role": "assistant", "content": reply, "reasoning_content": "hmm"}
			if len(req.Tools) > 0 {
				msg["tool_calls"] = []any{map[string]any{"index": 0, "id": "c1", "type": "function",
					"function": map[string]any{"name": "get_weather", "arguments": `{"city":"Lahore"}`}}}
			}
			json.NewEncoder(w).Encode(map[string]any{
				"choices": []any{map[string]any{"message": msg, "finish_reason": "stop"}},
				"usage":   map[string]int{"prompt_tokens": 7, "completion_tokens": 3},
			})
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		send := func(v any) { b, _ := json.Marshal(v); fmt.Fprintf(w, "data: %s\n\n", b); w.(http.Flusher).Flush() }
		send(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"reasoning_content": "hmm"}}}})
		for _, part := range []string{"echo: ", last} {
			send(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"content": part}}}})
		}
		if len(req.Tools) > 0 {
			send(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"tool_calls": []any{map[string]any{"index": 0, "function": map[string]any{"name": "get_weather", "arguments": `{"ci`}}}}}}})
			send(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"tool_calls": []any{map[string]any{"index": 0, "function": map[string]any{"arguments": `ty":"Lahore"}`}}}}}}})
		}
		send(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{}, "finish_reason": "stop"}}, "usage": map[string]int{"prompt_tokens": 7, "completion_tokens": 3}})
		fmt.Fprint(w, "data: [DONE]\n\n")
	})
	http.ListenAndServe(fmt.Sprintf("%s:%d", *host, *port), nil)
}
