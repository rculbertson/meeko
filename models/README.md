# Wake-word models

## `hey_meeko.onnx`

The wake-word model Meeko ships with. It listens for the phrase **"Hey Meeko"** and gates
the start of a session — until it fires, mic audio never reaches Deepgram.

- **Format:** ONNX, for [openWakeWord](https://github.com/dscripka/openWakeWord)'s ONNX
  inference backend (~415 KB).
- **How it was made:** trained with openWakeWord's own hosted training tool, which
  generates synthetic samples of the target phrase and trains a model from them. No
  recordings of anyone's voice were used, and none are embedded in the model.
- **Default threshold:** `0.96`, set in `[wake_word] threshold`. Lower it if Meeko doesn't
  hear you; raise it if it wakes on its own.

## Licensing

The model file is covered by this repository's MIT license. openWakeWord itself — the
runtime that loads this file, and the tooling that produced it — is Apache-2.0, which is
compatible.

openWakeWord also downloads its own shared preprocessor models (melspectrogram and
embedding ONNX files, ~3 MB) on first run. Those are not in this repo; they're fetched to
openWakeWord's cache. To pre-populate that cache on a network-connected machine before an
offline deploy:

```bash
uv run python -m meeko.wake_word
```

## Training your own wake word

Nothing about Meeko is tied to "Hey Meeko." To use a different phrase, train a model with
openWakeWord's training tooling, drop the resulting `.onnx` file in this directory, and
point Meeko at it:

```toml
[wake_word]
model = "models/hey_something_else.onnx"
```

Or set `MEEKO_WAKE_WORD_MODEL` for a one-off run. Expect to tune `threshold` for a new
model — the value that works for one phrase rarely transfers.

The `wake_word` key in each profile is only an advisory label used in prompts and logs; the
ONNX model is what actually determines the phrase Meeko listens for.
