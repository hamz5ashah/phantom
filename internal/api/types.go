// Package api holds Ollama-compatible request/response types and the
// translation to and from llama-server's OpenAI-style API.
package api

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Message is an Ollama chat message.
type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	Thinking  string     `json:"thinking,omitempty"`
	Images    []string   `json:"images,omitempty"` // base64
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	ToolName  string     `json:"tool_name,omitempty"`
}

// ToolCall is an Ollama tool call (arguments are an object, not a string).
type ToolCall struct {
	Function struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	} `json:"function"`
}

// Options are Ollama model options; only the common ones are used.
type Options struct {
	Temperature   *float64 `json:"temperature,omitempty"`
	TopP          *float64 `json:"top_p,omitempty"`
	TopK          *int     `json:"top_k,omitempty"`
	MinP          *float64 `json:"min_p,omitempty"`
	NumPredict    *int     `json:"num_predict,omitempty"`
	NumCtx        int      `json:"num_ctx,omitempty"`
	Seed          *int     `json:"seed,omitempty"`
	Stop          []string `json:"stop,omitempty"`
	RepeatPenalty *float64 `json:"repeat_penalty,omitempty"`
}

// ChatRequest is the body of POST /api/chat.
type ChatRequest struct {
	Model     string          `json:"model"`
	Messages  []Message       `json:"messages"`
	Stream    *bool           `json:"stream,omitempty"`
	Format    json.RawMessage `json:"format,omitempty"`
	Options   Options         `json:"options,omitempty"`
	Tools     json.RawMessage `json:"tools,omitempty"`
	Think     *bool           `json:"think,omitempty"`
	KeepAlive *Duration       `json:"keep_alive,omitempty"`
}

// GenerateRequest is the body of POST /api/generate.
type GenerateRequest struct {
	Model     string          `json:"model"`
	Prompt    string          `json:"prompt"`
	System    string          `json:"system,omitempty"`
	Images    []string        `json:"images,omitempty"`
	Stream    *bool           `json:"stream,omitempty"`
	Format    json.RawMessage `json:"format,omitempty"`
	Options   Options         `json:"options,omitempty"`
	Think     *bool           `json:"think,omitempty"`
	KeepAlive *Duration       `json:"keep_alive,omitempty"`
}

// ChatResponse is one line of the /api/chat stream (or the whole reply).
type ChatResponse struct {
	Model              string   `json:"model"`
	CreatedAt          string   `json:"created_at"`
	Message            *Message `json:"message,omitempty"`
	Response           *string  `json:"response,omitempty"` // /api/generate
	Thinking           string   `json:"thinking,omitempty"` // /api/generate
	Done               bool     `json:"done"`
	DoneReason         string   `json:"done_reason,omitempty"`
	TotalDuration      int64    `json:"total_duration,omitempty"`
	LoadDuration       int64    `json:"load_duration,omitempty"`
	PromptEvalCount    int      `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64    `json:"prompt_eval_duration,omitempty"`
	EvalCount          int      `json:"eval_count,omitempty"`
	EvalDuration       int64    `json:"eval_duration,omitempty"`
}

// Duration accepts Ollama's keep_alive forms: "5m", "1h", 300 (seconds), -1.
type Duration struct{ time.Duration }

func (d *Duration) UnmarshalJSON(b []byte) error {
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	switch t := v.(type) {
	case float64:
		if t < 0 {
			d.Duration = -1
		} else {
			d.Duration = time.Duration(t * float64(time.Second))
		}
	case string:
		if t == "" {
			return nil
		}
		if n, err := strconv.ParseFloat(t, 64); err == nil {
			return d.UnmarshalJSON([]byte(strconv.FormatFloat(n, 'f', -1, 64)))
		}
		p, err := time.ParseDuration(t)
		if err != nil {
			return fmt.Errorf("invalid keep_alive %q", t)
		}
		if p < 0 {
			p = -1
		}
		d.Duration = p
	default:
		return fmt.Errorf("invalid keep_alive %s", string(b))
	}
	return nil
}

// Now formats a timestamp like Ollama does.
func Now() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// --- OpenAI-style types used to talk to llama-server ---

type oaMessage struct {
	Role       string       `json:"role"`
	Content    any          `json:"content"`
	ToolCalls  []oaToolCall `json:"tool_calls,omitempty"`
	ToolCallID string       `json:"tool_call_id,omitempty"`
	Name       string       `json:"name,omitempty"`
}

type oaToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments,omitempty"`
	} `json:"function"`
}

// OARequest is the body sent to llama-server's /v1/chat/completions.
type OARequest struct {
	Messages        []oaMessage       `json:"messages"`
	Stream          bool              `json:"stream"`
	StreamOptions   map[string]any    `json:"stream_options,omitempty"`
	Temperature     *float64          `json:"temperature,omitempty"`
	TopP            *float64          `json:"top_p,omitempty"`
	TopK            *int              `json:"top_k,omitempty"`
	MinP            *float64          `json:"min_p,omitempty"`
	MaxTokens       *int              `json:"max_tokens,omitempty"`
	Seed            *int              `json:"seed,omitempty"`
	Stop            []string          `json:"stop,omitempty"`
	RepeatPenalty   *float64          `json:"repeat_penalty,omitempty"`
	Tools           json.RawMessage   `json:"tools,omitempty"`
	ResponseFormat  map[string]any    `json:"response_format,omitempty"`
	ChatTemplateKws map[string]any    `json:"chat_template_kwargs,omitempty"`
	Extra           map[string]string `json:"-"`
}

// ToOpenAI converts an Ollama chat request for llama-server.
func ToOpenAI(r *ChatRequest) (*OARequest, error) {
	out := &OARequest{Stream: r.Stream == nil || *r.Stream}
	if out.Stream {
		out.StreamOptions = map[string]any{"include_usage": true}
	}
	callID := 0
	for _, m := range r.Messages {
		om := oaMessage{Role: m.Role, Content: m.Content}
		if len(m.Images) > 0 {
			parts := []map[string]any{{"type": "text", "text": m.Content}}
			for _, img := range m.Images {
				parts = append(parts, map[string]any{
					"type":      "image_url",
					"image_url": map[string]string{"url": "data:" + imageMime(img) + ";base64," + img},
				})
			}
			om.Content = parts
		}
		for _, tc := range m.ToolCalls {
			args, err := json.Marshal(tc.Function.Arguments)
			if err != nil {
				return nil, err
			}
			callID++
			c := oaToolCall{ID: fmt.Sprintf("call_%d", callID), Type: "function"}
			c.Function.Name, c.Function.Arguments = tc.Function.Name, string(args)
			om.ToolCalls = append(om.ToolCalls, c)
		}
		if m.Role == "tool" {
			om.ToolCallID = fmt.Sprintf("call_%d", callID)
			om.Name = m.ToolName
		}
		out.Messages = append(out.Messages, om)
	}
	o := r.Options
	out.Temperature, out.TopP, out.TopK, out.MinP = o.Temperature, o.TopP, o.TopK, o.MinP
	out.Seed, out.Stop, out.RepeatPenalty = o.Seed, o.Stop, o.RepeatPenalty
	if o.NumPredict != nil && *o.NumPredict >= 0 {
		out.MaxTokens = o.NumPredict
	}
	if len(r.Tools) > 0 && string(r.Tools) != "null" {
		out.Tools = r.Tools
	}
	if f := strings.TrimSpace(string(r.Format)); f != "" && f != "null" && f != `""` {
		if f == `"json"` {
			out.ResponseFormat = map[string]any{"type": "json_object"}
		} else {
			var schema any
			if err := json.Unmarshal(r.Format, &schema); err != nil {
				return nil, fmt.Errorf("invalid format: %w", err)
			}
			out.ResponseFormat = map[string]any{"type": "json_schema", "json_schema": map[string]any{"name": "response", "schema": schema}}
		}
	}
	if r.Think != nil {
		out.ChatTemplateKws = map[string]any{"enable_thinking": *r.Think}
	}
	return out, nil
}

// FromGenerate turns /api/generate into a chat request.
func FromGenerate(g *GenerateRequest) *ChatRequest {
	c := &ChatRequest{Model: g.Model, Stream: g.Stream, Format: g.Format, Options: g.Options, Think: g.Think, KeepAlive: g.KeepAlive}
	if g.System != "" {
		c.Messages = append(c.Messages, Message{Role: "system", Content: g.System})
	}
	c.Messages = append(c.Messages, Message{Role: "user", Content: g.Prompt, Images: g.Images})
	return c
}

func imageMime(b64 string) string {
	switch {
	case strings.HasPrefix(b64, "iVBOR"):
		return "image/png"
	case strings.HasPrefix(b64, "R0lG"):
		return "image/gif"
	case strings.HasPrefix(b64, "UklG"):
		return "image/webp"
	default:
		return "image/jpeg"
	}
}

// --- llama-server responses ---

type oaChoice struct {
	Message      *oaDelta `json:"message"`
	Delta        *oaDelta `json:"delta"`
	FinishReason *string  `json:"finish_reason"`
}

type oaDelta struct {
	Content          *string      `json:"content"`
	ReasoningContent *string      `json:"reasoning_content"`
	ToolCalls        []oaToolCall `json:"tool_calls"`
}

type oaUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

type oaTimings struct {
	PromptMS    float64 `json:"prompt_ms"`
	PredictedMS float64 `json:"predicted_ms"`
}

// OAResponse is a llama-server chat completion (or one stream chunk).
type OAResponse struct {
	Choices []oaChoice `json:"choices"`
	Usage   *oaUsage   `json:"usage"`
	Timings *oaTimings `json:"timings"`
}

// Assembler turns llama-server output into Ollama responses, accumulating
// streamed tool-call fragments until they are complete.
type Assembler struct {
	Model    string
	Generate bool // shape lines as /api/generate responses
	Start    time.Time
	LoadDur  time.Duration

	calls    map[int]*oaToolCall
	order    []int
	finish   string
	usage    *oaUsage
	timings  *oaTimings
	content  strings.Builder
	thinking strings.Builder
}

// Chunk converts one streamed chunk; it returns nil when there is nothing to send yet.
func (a *Assembler) Chunk(c *OAResponse) *ChatResponse {
	a.note(c)
	if len(c.Choices) == 0 || c.Choices[0].Delta == nil {
		return nil
	}
	d := c.Choices[0].Delta
	for _, tc := range d.ToolCalls {
		a.addCall(tc)
	}
	text, think := deref(d.Content), deref(d.ReasoningContent)
	if text == "" && think == "" {
		return nil
	}
	return a.line(text, think, nil)
}

// Final is the closing "done" line of a stream.
func (a *Assembler) Final() *ChatResponse {
	r := a.line("", "", a.toolCalls())
	a.finishLine(r)
	return r
}

// Whole converts a non-streamed completion.
func (a *Assembler) Whole(c *OAResponse) *ChatResponse {
	a.note(c)
	text, think := "", ""
	if len(c.Choices) > 0 && c.Choices[0].Message != nil {
		m := c.Choices[0].Message
		text, think = deref(m.Content), deref(m.ReasoningContent)
		for _, tc := range m.ToolCalls {
			a.addCall(tc)
		}
	}
	r := a.line(text, think, a.toolCalls())
	a.finishLine(r)
	return r
}

func (a *Assembler) note(c *OAResponse) {
	if c.Usage != nil {
		a.usage = c.Usage
	}
	if c.Timings != nil {
		a.timings = c.Timings
	}
	if len(c.Choices) > 0 && c.Choices[0].FinishReason != nil && *c.Choices[0].FinishReason != "" {
		a.finish = *c.Choices[0].FinishReason
	}
}

func (a *Assembler) addCall(tc oaToolCall) {
	if a.calls == nil {
		a.calls = map[int]*oaToolCall{}
	}
	cur, ok := a.calls[tc.Index]
	if !ok {
		cp := tc
		a.calls[tc.Index] = &cp
		a.order = append(a.order, tc.Index)
		return
	}
	if tc.Function.Name != "" {
		cur.Function.Name = tc.Function.Name
	}
	cur.Function.Arguments += tc.Function.Arguments
}

func (a *Assembler) toolCalls() []ToolCall {
	var out []ToolCall
	for _, i := range a.order {
		c := a.calls[i]
		var tc ToolCall
		tc.Function.Name = c.Function.Name
		tc.Function.Arguments = map[string]any{}
		if s := strings.TrimSpace(c.Function.Arguments); s != "" {
			_ = json.Unmarshal([]byte(s), &tc.Function.Arguments)
		}
		out = append(out, tc)
	}
	return out
}

func (a *Assembler) line(text, think string, calls []ToolCall) *ChatResponse {
	a.content.WriteString(text)
	a.thinking.WriteString(think)
	r := &ChatResponse{Model: a.Model, CreatedAt: Now()}
	if a.Generate {
		r.Response, r.Thinking = &text, think
	} else {
		r.Message = &Message{Role: "assistant", Content: text, Thinking: think, ToolCalls: calls}
	}
	return r
}

func (a *Assembler) finishLine(r *ChatResponse) {
	r.Done = true
	r.DoneReason = "stop"
	switch a.finish {
	case "length":
		r.DoneReason = "length"
	case "tool_calls":
		r.DoneReason = "stop"
	}
	if !a.Start.IsZero() {
		r.TotalDuration = int64(time.Since(a.Start))
	}
	r.LoadDuration = int64(a.LoadDur)
	if a.usage != nil {
		r.PromptEvalCount, r.EvalCount = a.usage.PromptTokens, a.usage.CompletionTokens
	}
	if a.timings != nil {
		r.PromptEvalDuration = int64(a.timings.PromptMS * float64(time.Millisecond))
		r.EvalDuration = int64(a.timings.PredictedMS * float64(time.Millisecond))
	}
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}
