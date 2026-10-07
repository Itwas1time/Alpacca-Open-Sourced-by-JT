# Security and local data

Alpaccaroo is intended for personal local inference. The API server defaults to
127.0.0.1 and provides no authentication or TLS. Keep it on loopback; binding
to other interfaces, port forwarding, or publishing it through a tunnel lets
others use the model and consume machine resources. Do not expose it directly
to the internet.

Model downloads contact Ollama or Hugging Face. Optional package installation
also uses the internet. After setup, local model inference does not require
internet access. Treat model files and dependencies as untrusted inputs and
review their sources and licenses.

HF_TOKEN and HUGGING_FACE_HUB_TOKEN are read from your environment for gated
downloads. Supply your own credentials locally; never commit them. Local chat
history and models live under ALPACCAROO_HOME (normally ~/.alpaccaroo). Keep
that directory, private prompts, profiling output, and logs out of Git.
Do not post secrets or device details in public issues. Use GitHub private
vulnerability reporting for security reports.
