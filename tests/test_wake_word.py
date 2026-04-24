"""Unit tests for `meeko.wake_word.WakeWordDetector`.

openWakeWord's Model is replaced by a fake whose `predict` returns a
pre-scripted sequence of scores. This keeps the tests offline (no ONNX
load, no model file required) and lets us assert framing, threshold,
and buffer-carryover behavior directly.
"""

import numpy as np
import pytest

from meeko import wake_word
from meeko.wake_word import FRAME_BYTES, FRAME_SAMPLES, WakeWordDetector


class _FakeModel:
    def __init__(self, wakeword_models, inference_framework):
        self.wakeword_models = wakeword_models
        self.inference_framework = inference_framework
        self.predict_calls: list[np.ndarray] = []
        self.scores_queue: list[float] = []
        self.reset_calls = 0

    def predict(self, samples):
        self.predict_calls.append(samples)
        score = self.scores_queue.pop(0) if self.scores_queue else 0.0
        return {"hey_meeko": score}

    def reset(self):
        self.reset_calls += 1


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    model_path = tmp_path / "hey_meeko.onnx"
    model_path.write_bytes(b"stub")
    created: dict = {}

    def factory(wakeword_models, inference_framework):
        m = _FakeModel(wakeword_models, inference_framework)
        created["model"] = m
        return m

    monkeypatch.setattr(wake_word, "Model", factory)
    return str(model_path), created


def test_missing_model_file_raises(tmp_path):
    missing = tmp_path / "nope.onnx"
    with pytest.raises(FileNotFoundError):
        WakeWordDetector(model_path=str(missing))


def test_below_threshold_does_not_fire(fake_model):
    model_path, created = fake_model
    det = WakeWordDetector(model_path=model_path, threshold=0.5)
    created["model"].scores_queue = [0.1, 0.2, 0.4]

    # Three full frames — each evaluated, none crosses threshold.
    assert det.process(b"\x00\x00" * FRAME_SAMPLES) is False
    assert det.process(b"\x00\x00" * FRAME_SAMPLES) is False
    assert det.process(b"\x00\x00" * FRAME_SAMPLES) is False
    assert len(created["model"].predict_calls) == 3


def test_fires_when_score_crosses_threshold(fake_model):
    model_path, created = fake_model
    det = WakeWordDetector(model_path=model_path, threshold=0.5)
    created["model"].scores_queue = [0.1, 0.9]

    assert det.process(b"\x00\x00" * FRAME_SAMPLES) is False
    assert det.process(b"\x00\x00" * FRAME_SAMPLES) is True


def test_buffers_partial_chunks_into_full_frames(fake_model):
    """50ms mic chunks (800 samples) must accumulate into 1280-sample
    frames before predict is called."""
    model_path, created = fake_model
    det = WakeWordDetector(model_path=model_path, threshold=0.5)
    # Two mic chunks (1600 samples) — enough for exactly one frame, with
    # 320 samples carried over.
    mic_chunk = b"\x00\x00" * 800

    assert det.process(mic_chunk) is False
    assert len(created["model"].predict_calls) == 0, (
        "first 800-sample chunk should not trigger predict"
    )

    created["model"].scores_queue = [0.9]
    assert det.process(mic_chunk) is True
    assert len(created["model"].predict_calls) == 1
    # The frame passed to predict is exactly 1280 int16 samples.
    assert created["model"].predict_calls[0].shape == (FRAME_SAMPLES,)
    assert created["model"].predict_calls[0].dtype == np.int16


def test_multiple_frames_per_chunk_all_evaluated(fake_model):
    """A chunk containing multiple full frames should evaluate each
    until one fires or all are consumed."""
    model_path, created = fake_model
    det = WakeWordDetector(model_path=model_path, threshold=0.5)
    created["model"].scores_queue = [0.1, 0.2, 0.9, 0.0]

    # 3 frames in one big chunk; third frame fires.
    big = b"\x00\x00" * (FRAME_SAMPLES * 3)
    assert det.process(big) is True
    assert len(created["model"].predict_calls) == 3


def test_detection_clears_buffer(fake_model):
    model_path, created = fake_model
    det = WakeWordDetector(model_path=model_path, threshold=0.5)
    created["model"].scores_queue = [0.9]

    # One frame + partial; detection clears everything.
    assert det.process(b"\x00\x00" * (FRAME_SAMPLES + 100)) is True
    assert len(det._buffer) == 0


def test_unknown_score_key_falls_back_to_first_value(fake_model, monkeypatch):
    """If the model returns a differently-named score key we still pick
    it up instead of silently treating it as zero."""
    model_path, _ = fake_model

    class _OddKeyModel:
        def __init__(self, wakeword_models, inference_framework):
            pass

        def predict(self, samples):
            return {"something_else": 0.99}

    monkeypatch.setattr(wake_word, "Model", _OddKeyModel)
    det = WakeWordDetector(model_path=model_path, threshold=0.5)
    assert det.process(b"\x00\x00" * FRAME_SAMPLES) is True


def test_reset_clears_buffer_and_model_state(fake_model):
    """Both the frame-alignment buffer AND openWakeWord's internal
    prediction / preprocessor state must be cleared on re-arm. Without
    the model-side reset, the just-fired wake context carries into the
    new IDLE window and can immediately re-trigger."""
    model_path, created = fake_model
    det = WakeWordDetector(model_path=model_path, threshold=0.5)

    # Leave a partial frame in the buffer and bump the model's reset
    # counter's baseline.
    det.process(b"\x00\x00" * 100)
    assert len(det._buffer) > 0
    baseline_resets = created["model"].reset_calls

    det.reset()

    assert len(det._buffer) == 0
    assert created["model"].reset_calls == baseline_resets + 1


def test_default_model_path():
    assert wake_word.default_model_path().endswith("hey_meeko.onnx")


def test_frame_constants():
    # 80ms @ 16kHz = 1280 samples, 2560 bytes (int16 mono).
    assert FRAME_SAMPLES == 1280
    assert FRAME_BYTES == 2560


def test_preprocessor_download_failure_raises_operator_friendly_error(
    monkeypatch, tmp_path
):
    """If the preprocessor files are missing and the download fails
    (e.g. offline host), WakeWordDetector should raise a RuntimeError
    that points the operator at the pre-fetch command, with the
    underlying exception preserved on __cause__."""
    import os as _os

    model_path = tmp_path / "hey_meeko.onnx"
    model_path.write_bytes(b"stub")

    real_exists = _os.path.exists

    def fake_exists(p):
        # Pretend the preprocessor cache is empty but the wake-word
        # model file is still present so the missing-file guard passes.
        if p.endswith(("melspectrogram.onnx", "embedding_model.onnx")):
            return False
        return real_exists(p)

    original_cause = OSError("network down")

    def fail_download():
        raise original_cause

    monkeypatch.setattr(wake_word.os.path, "exists", fake_exists)
    monkeypatch.setattr(wake_word.openwakeword.utils, "download_models", fail_download)

    with pytest.raises(RuntimeError) as excinfo:
        WakeWordDetector(model_path=str(model_path))
    assert "python -m meeko.wake_word" in str(excinfo.value)
    assert excinfo.value.__cause__ is original_cause
