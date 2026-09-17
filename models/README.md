# Wake-word models

## `hey_meeko.onnx`

The wake-word model Meeko ships with. It listens for the phrase **"Hey Meeko"** and gates
the start of a session: while the gate is armed, mic audio is fed to the local detector
only and never reaches Deepgram. (Setting `[wake_word] disabled = true` bypasses the gate
entirely and streams to Deepgram from the first frame — it exists for development.)

- **Format:** ONNX, for [openWakeWord](https://github.com/dscripka/openWakeWord)'s ONNX
  inference backend (~415 KB).
- **How it was made:** trained with openWakeWord's own hosted training tool, which
  generates synthetic samples of the target phrase and trains a model from them. No
  recordings of anyone's voice were used, and none are embedded in the model.
- **Default threshold:** `0.96`, set in `[wake_word] threshold`. Lower it if Meeko doesn't
  hear you; raise it if it wakes on its own.

## Licensing

**Read this before shipping anything commercial.**

`hey_meeko.onnx` itself is covered by this repository's MIT license, and openWakeWord's
*code* is Apache-2.0. But openWakeWord's **pre-trained models are CC-BY-NC-SA-4.0**
(NonCommercial, ShareAlike) — upstream's wording: *"All of the included pre-trained models
are licensed under [CC-BY-NC-SA-4.0] due to the inclusion of datasets with unknown or
restrictive licensing as part of the training data."*

That matters here because the shared preprocessor models Meeko loads on every wake-word
run — `melspectrogram.onnx` and `embedding_model.onnx` — are those pre-trained models.
They are downloaded to openWakeWord's own cache on first run, not vendored in this repo,
but the wake-word path depends on them. So while Meeko's source is MIT, **the default
wake-word path is not usable in a commercial product** without replacing those
preprocessors or obtaining different terms. Running Meeko personally is unaffected.

`[wake_word] disabled = true` skips the wake-word path entirely and touches none of these
files.

## First-run download

On first run openWakeWord fetches its models into its own cache. Meeko calls
`download_models()` with no arguments, which pulls **openWakeWord's full model set (~19 MB
measured)** — the two preprocessors Meeko actually uses total about 2.4 MB; the rest are
bundled wake words (`alexa`, `hey_jarvis`, `hey_mycroft`, `hey_rhasspy`, `timer`,
`weather`) and `silero_vad`, none of which Meeko loads. Budget accordingly when baking an
offline Pi image. To pre-populate the cache on a network-connected machine:

```bash
uv run python -m meeko.wake_word
```

## Training your own wake word

Nothing about Meeko is tied to "Hey Meeko." To use a different phrase, train a model with
openWakeWord's training tooling, drop the resulting `.onnx` file in this directory, and
point Meeko at it:

```toml
[wake_word]
model = "/home/you/meeko/models/hey_something_else.onnx"
```

Use an absolute path (or one starting `~/`). The value is resolved relative to the process
working directory, not to the config file — and the config normally lives at
`~/.config/meeko/meeko.toml`, so a relative path like `models/…` only works when Meeko is
launched from the repo root. Or set `MEEKO_WAKE_WORD_MODEL` for a one-off run. Expect to tune `threshold` for a new
model — the value that works for one phrase rarely transfers.

The `wake_word` key in each profile is only an advisory label used in prompts and logs; the
ONNX model is what actually determines the phrase Meeko listens for.
