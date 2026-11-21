# Third-party notices

## llama.cpp

Phantom runs models with `llama-server` from llama.cpp, which `phantom setup`
downloads from the official releases at https://github.com/ggml-org/llama.cpp.
It is not part of Phantom's source code. If you ship it with Phantom, include
this notice:

    MIT License

    Copyright (c) 2023-2026 The ggml authors

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## Models

Models downloaded from the Ollama library or Hugging Face are made by their
authors and come with their own licenses (for example Llama, Gemma, Qwen,
Mistral). Check a model's license before using it commercially.

## API compatibility

Phantom implements an HTTP API compatible with Ollama's so existing apps work
with it. Phantom is not affiliated with or endorsed by Ollama.
