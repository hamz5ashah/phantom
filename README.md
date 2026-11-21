# Phantom

Run open AI models on your own computer. Phantom downloads models from the
**Ollama library** or **Hugging Face**, runs them with llama.cpp on your CPU or
GPU, and serves an **Ollama-compatible API**, so apps built for Ollama
(including Ghost) work with it unchanged. It also speaks the OpenAI API.

## Quick start

```bash
phantom run qwen3:0.6b
```

The first time, Phantom downloads llama.cpp for your PC and then the model.
After that everything runs offline.

```bash
phantom serve                         # start the API on http://127.0.0.1:11435
phantom pull gemma3:4b                # Ollama library model
phantom pull hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M   # any Hugging Face GGUF
phantom list                          # downloaded models
phantom ps                            # loaded models
phantom rm gemma3:4b
phantom setup --backend cpu           # reinstall llama.cpp (vulkan | cpu | cuda | metal)
```

> **Model support:** any GGUF model llama.cpp supports works, and most Ollama
> library models are standard GGUF files. A few of Ollama's newest models are
> packaged for Ollama's own engine and may not load; use the Hugging Face GGUF of
> the same model instead (`hf.co/...`).

## API

Ollama-compatible (`/api/...`): `chat`, `generate`, `embed`, `embeddings`, `pull`,
`tags`, `show`, `ps`, `delete`, `version`, with streaming, thinking, tool calls, images
(vision models), JSON output and `keep_alive`.

OpenAI-compatible (`/v1/...`): `chat/completions`, `completions`, `embeddings`,
`models`.

```bash
curl http://127.0.0.1:11435/api/chat -d '{
  "model": "qwen3:0.6b",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

To point an Ollama app at Phantom, set its Ollama address to
`http://127.0.0.1:11435`, or run Phantom on Ollama's port with
`PHANTOM_HOST=127.0.0.1:11434`.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `PHANTOM_HOST` | `127.0.0.1:11435` | Address to listen on / connect to |
| `PHANTOM_MODELS` | `~/.ollama/models` if it exists, else `~/.phantom/models` | Where models are stored (models downloaded with Ollama are reused) |
| `PHANTOM_KEEP_ALIVE` | `5m` | How long an idle model stays in memory |
| `PHANTOM_MAX_LOADED` | `1` | Models kept in memory at once |
| `PHANTOM_LLAMA_SERVER` | | Use your own `llama-server` build |
| `PHANTOM_DEBUG` | | Show llama.cpp's output |

## How it works

- `internal/registry`: downloads models from Ollama-compatible registries
  (`registry.ollama.ai`, `hf.co`), resumable, with sha256 checks.
- `internal/store`: keeps models on disk (`blobs/` + `manifests/`).
- `internal/runner`: starts one `llama-server` per loaded model with the GGUF's
  own chat template, offloads to the GPU, and unloads idle models.
- `internal/server`: the HTTP API; translates Ollama requests to llama.cpp.
- `internal/setup`: downloads the right official llama.cpp build for your PC.

## Building

```bash
go build -o phantom .
go test ./...
```

Releases for Windows, Linux and macOS are built by GitHub Actions when a `v*`
tag is pushed, or from Actions → Build → Run workflow.

## License

Proprietary, see [LICENSE](LICENSE). llama.cpp and the models keep their own
licenses, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
