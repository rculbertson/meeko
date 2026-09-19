"""openWakeWord integration.

Runs a trained ONNX wake-word model on streaming mic audio and reports
the first frame whose confidence exceeds a threshold. Used to gate the
orchestrator's entry from IDLE into LISTENING.

openWakeWord's native hop is 80ms (1280 samples at 16 kHz, int16 mono).
Our mic delivers 50ms chunks; this class buffers incoming chunks and
runs inference on each 1280-sample frame as it becomes available.

Inference on the ~415 KB Hey-Meeko model is well under 1 ms per frame
on both Mac and Pi 5, so `process()` is safe to call inline on the
asyncio event loop without offloading to a thread.
"""

import logging
import os

import numpy as np
import openwakeword.utils
from openwakeword.model import Model

logger = logging.getLogger("meeko")

# openWakeWord expects frames of 1280 int16 samples at 16 kHz (80 ms).
FRAME_SAMPLES = 1280
FRAME_BYTES = FRAME_SAMPLES * 2


def default_model_path() -> str:
    return os.path.join("models", "hey_meeko.onnx")


def _preprocessor_cache_dir() -> str:
    return os.path.join(
        os.path.dirname(openwakeword.utils.__file__), "resources", "models"
    )


# openWakeWord's download_models() always fetches the melspectrogram,
# embedding and VAD models — the only ones Meeko loads. With an empty
# model_names it *also* pulls its six bundled wake words (alexa, hey_jarvis,
# hey_mycroft, hey_rhasspy, timer, weather), ~12 MB Meeko never touches.
#
# Passing any non-empty list that matches no official model name suppresses
# those extras, cutting the first-run download from ~19 MB to ~6.7 MB.
# Matching is a substring test against official filenames, so both [] and
# [""] would pull the full set — the value has to be non-empty and specific.
_NO_BUNDLED_MODELS = ["hey_meeko"]


def _ensure_preprocessors() -> None:
    """Fetch openWakeWord's shared melspectrogram + embedding ONNX
    preprocessors on first run. Idempotent — no-op if the files are
    already cached. Raises RuntimeError with an operator-friendly
    remediation hint if the download fails (e.g. offline host)."""
    resources = _preprocessor_cache_dir()
    melspec = os.path.join(resources, "melspectrogram.onnx")
    embedding = os.path.join(resources, "embedding_model.onnx")
    if os.path.exists(melspec) and os.path.exists(embedding):
        return
    logger.info("Downloading openWakeWord preprocessor models (one-time)")
    try:
        openwakeword.utils.download_models(model_names=_NO_BUNDLED_MODELS)
    except Exception as exc:
        raise RuntimeError(
            "Failed to download openWakeWord preprocessor models. Run "
            "`uv run python -m meeko.wake_word` from a machine with network "
            "access to pre-fetch them."
        ) from exc


class WakeWordDetector:
    def __init__(
        self,
        threshold: float,
        model_path: str | None = None,
    ):
        path = model_path or default_model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Wake-word model not found at {path!r}")
        self._threshold = threshold
        _ensure_preprocessors()
        self._model = Model(
            wakeword_models=[path],
            inference_framework="onnx",
        )
        # Name under which this model's score is returned by predict().
        # Model indexes by the file stem (e.g. "hey_meeko").
        self._score_key = os.path.splitext(os.path.basename(path))[0]
        self._buffer = bytearray()
        logger.info(
            "Wake-word detector ready (model=%s, threshold=%.2f)",
            path,
            threshold,
        )

    def reset(self) -> None:
        """Clear buffered audio AND the model's internal prediction /
        feature state. Called when re-arming the detector after a
        session ends.

        Clearing only our own frame buffer is not enough — openWakeWord's
        `Model` keeps a rolling prediction deque and preprocessor feature
        history across `predict()` calls. Without resetting those, the
        model still "remembers" the fire that started the just-ended
        session, and the first frame of the new IDLE window can re-cross
        threshold immediately on silence or low-level noise."""
        self._buffer.clear()
        self._model.reset()

    def process(self, pcm: bytes) -> bool:
        """Feed a mono 16-bit PCM chunk. Returns True on wake detection.

        Accepts chunks of any length — internal buffering handles the
        mismatch between our 50ms mic chunks and openWakeWord's 80ms
        frames.
        """
        self._buffer.extend(pcm)
        while len(self._buffer) >= FRAME_BYTES:
            frame_bytes = bytes(self._buffer[:FRAME_BYTES])
            del self._buffer[:FRAME_BYTES]
            samples = np.frombuffer(frame_bytes, dtype=np.int16)
            scores = self._model.predict(samples)
            score = scores.get(self._score_key)
            if score is None:
                # Fall back to whichever score the model returned —
                # handles unexpected key naming without silently failing.
                score = next(iter(scores.values()), 0.0)
            if score >= self._threshold:
                logger.info("Wake word detected (score=%.3f)", score)
                self._buffer.clear()
                return True
        return False


if __name__ == "__main__":
    # Populate the openWakeWord preprocessor cache out-of-band — useful
    # when baking a Pi image on a network-connected host before
    # deploying to a device without internet.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _ensure_preprocessors()
    print(f"openWakeWord preprocessors cached at: {_preprocessor_cache_dir()}")
