# Third-party notices

Alpaccaroo contains **no third-party code**: every module in `alpaccaroo/` is an
original Python implementation, and the runtime depends only on the Python
standard library (NumPy is an optional, auto-detected accelerator and is
never required or fetched).

It does, however, *interoperate* with file formats, protocols and
algorithms designed by others. Credit where it's due:

## GGUF format & quantization schemes

`alpaccaroo/gguf.py` and `alpaccaroo/quants.py` implement, from the published
specification and format documentation, the GGUF model file format and its
quantization block formats (Q4_0 ... Q6_K). GGUF and these schemes were
designed by the **ggml / llama.cpp project** (MIT, (c) The ggml authors,
<https://github.com/ggml-org/llama.cpp>). No ggml or llama.cpp source code
is included or linked.

## Ollama registry protocol

`alpaccaroo/pull.py` speaks the public model-distribution protocol of
**Ollama** (MIT, (c) Ollama Inc., <https://github.com/ollama/ollama>) -
OCI-style manifests with content-addressed layers served from
`registry.ollama.ai` - and Alpaccaroo's model-management UX (`pull` / `run` /
`list` / `rm`, `name:tag` references) is openly inspired by Ollama's. The
client is an independent implementation; no Ollama source code is included.

## Tokenization algorithms

`alpaccaroo/tokenizer.py` implements the SentencePiece-style subword
segmentation used by llama-family models (Kudo & Richardson, 2018) and
byte-level BPE with a GPT-2 style pre-tokenizer (Radford et al., 2019;
extended by Llama 3). These are published algorithms; the implementations
here are original.

## Hugging Face Hub API

`alpaccaroo/pull.py` uses Hugging Face's public HTTP API
(<https://huggingface.co>) to list and download community-hosted GGUF
files. No Hugging Face libraries are included.

## Models

Model weights are not part of Alpaccaroo. Every model you `alpaccaroo pull`
carries its publisher's own license; when the source provides a license
file, Alpaccaroo stores it next to the weights (see `alpaccaroo show <model>`).
Review it before redistribution or commercial use.
