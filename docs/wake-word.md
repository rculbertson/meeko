# Wake-Word Models & Training

Meeko uses [openWakeWord](https://github.com/dscripka/openWakeWord) for efficient, on-device wake-word detection. Audio is continuously processed locally until the wake phrase is detected, gating any streaming to Deepgram speech-to-text.

---

## Bundled Model: `hey_meeko.onnx`

Meeko ships with a pre-trained model for the wake phrase **"Hey Meeko"**, located at `models/hey_meeko.onnx`.

- **Format**: ONNX model optimized for openWakeWord's ONNX Runtime inference backend (~415 KB).
- **Training Method**: Trained using openWakeWord's automated synthetic training pipeline, which generates synthetic audio samples of the target phrase across diverse synthetic voices, room acoustics, and noise profiles. No recordings of real human voices were collected or embedded in the model.
- **Default Threshold**: `0.96` (configured in `[wake_word] threshold` in `meeko.toml`). Lower values make detection more sensitive at the cost of potential false wakes; higher values require clearer pronunciation.

To bypass wake-word detection entirely (e.g., during development), set `disabled = true` under `[wake_word]` in `meeko.toml`.

---

## Commercial Licensing Notice

> [!IMPORTANT]
> **Read this before using Meeko in a commercial product.**

While Meeko's own source code is licensed under the **MIT License** and openWakeWord's code is licensed under **Apache-2.0**, both the bundled `models/hey_meeko.onnx` model (generated via openWakeWord's training pipeline) and openWakeWord's **pre-trained preprocessor models are licensed under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0):

> *"All of the included pre-trained models are licensed under [CC-BY-NC-SA-4.0] due to the inclusion of datasets with unknown or restrictive licensing as part of the training data."*

Both `hey_meeko.onnx` and the shared preprocessor models downloaded on first run (`melspectrogram.onnx`, `embedding_model.onnx`) are subject to this NonCommercial restriction.

Therefore:
- **Personal and non-commercial use** is completely unrestricted.
- **Commercial use** requires replacing those preprocessor models with commercially licensed alternatives or obtaining separate commercial licensing terms from openWakeWord's maintainers.

Setting `[wake_word] disabled = true` skips the wake-word engine completely and touches none of these files.

---

## First-Run Model Download & Offline Caching

On its first run, openWakeWord automatically downloads its core preprocessor models (~6.7 MB) into its local cache:
- `melspectrogram.onnx`
- `embedding_model.onnx`
- `silero_vad.onnx`

*(Note: openWakeWord by default attempts to download six bundled wake-word models totaling ~12 MB (`alexa`, `hey_jarvis`, `hey_mycroft`, `hey_rhasspy`, `timer`, `weather`). Meeko's `_ensure_preprocessors()` explicitly suppresses those unused downloads).*

### Pre-populating Cache for Offline Devices

If you are deploying Meeko to a Raspberry Pi or other device with no internet access (or baking a disk image), pre-populate the openWakeWord cache by running the downloader on a connected machine:

```bash
uv run python -m meeko.wake_word
```

---

## Training a Custom Wake Word

Nothing about Meeko is hardcoded to "Hey Meeko." You can train a custom model for any phrase of your choice using openWakeWord's training pipeline:

1. **Train the Model**: Follow [openWakeWord's training documentation or Google Colab notebook](https://github.com/dscripka/openWakeWord#training-new-models) to generate a custom `.onnx` model file for your chosen phrase.
2. **Copy the Model**: Place the generated `.onnx` file in Meeko's `models/` directory (e.g., `models/hey_computer.onnx`).
3. **Update Configuration**: Update `meeko.toml` to point to the new model:

   ```toml
   [wake_word]
   model = "models/hey_computer.onnx"
   threshold = 0.85
   ```

   > [!TIP]
   > When running Meeko as a background service or outside the repo root, use an absolute path for `model` (e.g., `/home/pi/meeko/models/hey_computer.onnx`).
   >
   > Expect to tune `threshold` for a newly trained model; optimal thresholds vary depending on phrase length and phonetics.
