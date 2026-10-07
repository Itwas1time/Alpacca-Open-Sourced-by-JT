# Alpacca / Alpaccaroo

Alpaccaroo runs GGUF language models locally in Python. The installed module
and command are named `alpaccaroo`. Python 3.10 or newer is required; the
standard-library backend works without optional acceleration packages.

## Install and run

```sh
git clone https://github.com/Itwas1time/Alpacca.git
cd Alpacca
python -m alpaccaroo doctor
python -m alpaccaroo run /path/to/model.gguf
```

To install a command launcher, use `scripts/install.sh` on Linux/macOS or
`scripts/install.ps1` in Windows PowerShell. A normal Python installation with
`python -m pip install .` also works. Optional acceleration: `python -m pip
install ".[fast]"` for NumPy or `".[kernels]"` for the pinned CPU kernel tier.
GPU extras require compatible hardware and drivers; inspect `pyproject.toml`
before installing them. Pure Python can be slow and memory hungry on large models.

## Models, chat, and local storage

```sh
alpaccaroo pull llama3.2:1b
alpaccaroo list
alpaccaroo menu
alpaccaroo run llama3.2:1b "Explain a rainbow"
alpaccaroo history list
alpaccaroo history stats
```

`pull` downloads from Ollama or Hugging Face; `run` can download a missing
model. Use a local GGUF path for offline inference. Models, settings, and
interactive chat history live in `~/.alpaccaroo`, or `ALPACCAROO_HOME`.
One-shot prompts and API requests are not saved as interactive history.
Use `alpaccaroo --help` and each command's `--help` for model management,
context limits, sampling, benchmarking, profiling, and tuning options.
Architecture and quantization support varies; unsupported models may fail.
Model weights are supplied separately and have their own licenses.

## Local API and security

```sh
alpaccaroo serve /path/to/model.gguf --host 127.0.0.1 --port 8080
```

The OpenAI-compatible endpoint is `http://127.0.0.1:8080/v1`. The server has
no authentication or TLS. Keep it on loopback and do not expose it through
router forwarding or an internet tunnel. Network clients can consume model
resources and submit prompts if you bind it to a reachable interface.
Keep chat history, tokens, models, and local logs out of repositories.
Downloads and dependency installation use the internet; inference with an
already available local model does not. See [SECURITY.md](SECURITY.md) and
[third-party notices](THIRD-PARTY-NOTICES.md).
