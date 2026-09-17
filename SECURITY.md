# Security Policy

## Reporting a vulnerability

Please report security issues privately via [GitHub's private vulnerability reporting](https://github.com/rculbertson/meeko/security/advisories/new) rather than opening a public issue.

Meeko is maintained in spare time, so there is no response-time guarantee. Reports are read and taken seriously, but a fix may take a while. If something is actively dangerous to users, say so prominently in the report and it will be prioritized.

## Scope

Meeko is a single-user, single-device assistant that runs on your own hardware with your own API keys. The things most worth reporting:

- Anything that could leak `DEEPGRAM_API_KEY` or `ANTHROPIC_API_KEY` — for example, keys appearing in logs, in the SQLite database, or in an outbound request to anywhere other than Deepgram or Anthropic.
- Anything that exposes stored conversation transcripts beyond the local machine.
- Injection or path-traversal issues in the tool handlers under `meeko/tools/`, which act on model-supplied arguments.

## Not in scope

- The fact that audio and conversation text are sent to Deepgram and Anthropic. That is how Meeko works, and it is documented in the README under "Privacy".
- Physical access to the device. Transcripts are stored unencrypted in SQLite by design; anyone with your machine can read them.
- Your own API keys leaking through your own `.env` file, shell history, or backups.
