---
name: Bug report
about: Something isn't working
labels: bug
---

**What happened**

<!-- What you said or did, and what Meeko did instead. -->

**What you expected**

**Setup**

- Platform: <!-- Raspberry Pi 5 / macOS / other -->
- Audio device: <!-- ReSpeaker XVF3800 / built-in mic / other -->
- Python version: <!-- uv run python --version -->
- Meeko commit: <!-- git rev-parse --short HEAD -->

**Relevant config**

<!-- The relevant part of your meeko.toml, with [location] coordinates removed. -->

**Logs**

<!--
The default log_level is INFO (which logs state transitions, tool names, and errors without conversation content).
To capture debug logs containing detailed operational traces or transcripts, run with:
MEEKO_LOG_LEVEL=DEBUG MEEKO_LOG_TARGET=file uv run python -m meeko.main

Meeko will write meeko.log in the working directory.
NOTE: DEBUG logs contain transcripts of what you said — please trim any private or sensitive content before pasting.
-->
