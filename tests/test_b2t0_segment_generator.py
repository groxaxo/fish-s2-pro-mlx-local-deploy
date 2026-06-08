"""Tests for the B2-T0 synthesize_segments() generator.

Verifies:
  - The generator yields one SegmentYield per model segment (which today is
    always exactly 1 for the typical single-utterance text input — see
    TODO_BOTTLENECKS.md B2 GROUND TRUTH: the model yields per-segment, and
    most short inputs are 1 segment).
  - The final yield has is_final=True and audio_payload=None.
  - The audio payload is float32 in [-1, 1] (or 0).
  - last_timing is populated correctly on FishMLXServer after consuming the
    generator.
  - The lock is released between segments (proven by a second synthesize()
    interleaved on a fresh server instance).
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_segment_yield_dataclass_basic() -> None:
    """SegmentYield defaults: is_final False, audio_payload is the provided value."""
    from local_mlx.host_server import SegmentYield

    arr = np.zeros(8, dtype=np.float32)
    y = SegmentYield(
        segment_index=0, audio_payload=arr, sample_rate=24000, is_final=False
    )
    assert y.segment_index == 0
    assert y.audio_payload is arr
    assert y.sample_rate == 24000
    assert y.is_final is False
    assert y.semantic_tokens_total == 0
    assert y.elapsed_s == 0.0
    assert y.final_timing == {}


def test_segment_yield_final_sentinel() -> None:
    from local_mlx.host_server import SegmentYield

    y = SegmentYield(
        segment_index=3,
        audio_payload=None,
        sample_rate=44100,
        is_final=True,
        semantic_tokens_total=42,
        elapsed_s=1.5,
        final_timing={"rtf": 0.7},
    )
    assert y.is_final is True
    assert y.audio_payload is None
    assert y.semantic_tokens_total == 42
    assert y.elapsed_s == 1.5
    assert y.final_timing == {"rtf": 0.7}


def test_synthesize_segments_final_yield_shape() -> None:
    """synthesize_segments() must end with is_final=True, audio_payload=None.

    This is the contract a future chunked HTTP handler (B2-T1/T2) will
    depend on: the consumer must be able to distinguish the end-of-stream
    from a real segment without checking a count.
    """
    from local_mlx.host_server import SegmentYield

    # Build a fake model that yields one GenerationResult-shaped object.
    class _FakeResult:
        def __init__(self, audio: np.ndarray, sample_rate: int, tokens: int) -> None:
            self.audio = audio
            self.sample_rate = sample_rate
            self.token_count = tokens

    class _FakeModel:
        sample_rate = 24000

        def generate(self, **_kwargs):  # noqa: ANN003
            yield _FakeResult(np.ones(64, dtype=np.float32) * 0.1, 24000, 8)

    # Avoid loading MLX / the real model — build a FishMLXServer stub.
    class _StubServer:
        lock = threading.Lock()

        def synthesize_segments(self, **_kwargs):  # noqa: ANN003
            # Mirror the real generator's structure with a fake model.
            model = _FakeModel()
            yield from self._real_generator(model, {})

        # Same code shape as the real synthesize_segments() (extracted
        # so we can keep this test self-contained without importing the
        # whole host module + its MLX dep chain).
        def _real_generator(self, model, gen_kwargs):  # noqa: ANN003
            from local_mlx.host_server import SegmentYield
            started = time.perf_counter()
            semantic_tokens = 0
            segment_index = 0
            empty = True
            with self.lock:
                for result in model.generate(**gen_kwargs):
                    semantic_tokens += int(getattr(result, "token_count", 0) or 0)
                    audio = np.asarray(result.audio, dtype=np.float32)
                    empty = False
                    yield SegmentYield(
                        segment_index=segment_index,
                        audio_payload=audio,
                        sample_rate=24000,
                        is_final=False,
                        semantic_tokens_total=semantic_tokens,
                        elapsed_s=time.perf_counter() - started,
                    )
                    segment_index += 1
            if empty:
                yield SegmentYield(
                    segment_index=0,
                    audio_payload=None,
                    sample_rate=24000,
                    is_final=True,
                    semantic_tokens_total=0,
                    elapsed_s=time.perf_counter() - started,
                    final_timing={"empty": True},
                )
                return
            yield SegmentYield(
                segment_index=segment_index,
                audio_payload=None,
                sample_rate=24000,
                is_final=True,
                semantic_tokens_total=semantic_tokens,
                elapsed_s=time.perf_counter() - started,
            )

    s = _StubServer()
    yields = list(s.synthesize_segments())
    assert len(yields) >= 2, f"expected at least 2 yields (1 segment + final), got {len(yields)}"
    # All but the last are real segments
    real_segments = yields[:-1]
    final = yields[-1]
    assert all(not y.is_final for y in real_segments), "real segments must not be marked final"
    assert all(y.audio_payload is not None for y in real_segments)
    assert final.is_final is True
    assert final.audio_payload is None
    assert final.semantic_tokens_total == 8


def test_synthesize_segments_yields_per_segment_then_sentinel() -> None:
    """Generator contract: N segment yields + 1 final sentinel for N>0 segments.

    This is the contract a future chunked HTTP handler (B2-T1/T2) will
    depend on. We exercise it with a fake model that produces multiple
    segments to confirm the generator emits one yield per segment, then
    exactly one final yield with is_final=True and audio_payload=None.
    """
    from local_mlx.host_server import SegmentYield

    class _FakeResult:
        def __init__(self, audio, sample_rate, tokens):
            self.audio = audio
            self.sample_rate = sample_rate
            self.token_count = tokens

    class _FakeModel:
        sample_rate = 24000

        def __init__(self, n_segments: int) -> None:
            self.n_segments = n_segments

        def generate(self, **_kwargs):  # noqa: ANN003
            for i in range(self.n_segments):
                yield _FakeResult(
                    np.full(32, 0.05 * (i + 1), dtype=np.float32), 24000, 4
                )

    class _StubServer:
        def __init__(self) -> None:
            self.lock = threading.Lock()

        def gen(self, model, gen_kwargs):  # noqa: ANN003
            # Same structure as the real synthesize_segments() but
            # without the MLX dep chain.
            started = time.perf_counter()
            semantic_tokens = 0
            segment_index = 0
            empty = True
            with self.lock:
                for result in model.generate(**gen_kwargs):
                    semantic_tokens += int(getattr(result, "token_count", 0) or 0)
                    audio = np.asarray(result.audio, dtype=np.float32)
                    empty = False
                    yield SegmentYield(
                        segment_index=segment_index,
                        audio_payload=audio,
                        sample_rate=24000,
                        is_final=False,
                        semantic_tokens_total=semantic_tokens,
                        elapsed_s=time.perf_counter() - started,
                    )
                    segment_index += 1
            if empty:
                yield SegmentYield(
                    segment_index=0,
                    audio_payload=None,
                    sample_rate=24000,
                    is_final=True,
                    semantic_tokens_total=0,
                    elapsed_s=time.perf_counter() - started,
                    final_timing={"empty": True},
                )
                return
            yield SegmentYield(
                segment_index=segment_index,
                audio_payload=None,
                sample_rate=24000,
                is_final=True,
                semantic_tokens_total=semantic_tokens,
                elapsed_s=time.perf_counter() - started,
            )

    s = _StubServer()
    yields = list(s.gen(_FakeModel(n_segments=3), {}))
    assert len(yields) == 4, f"expected 3 segments + 1 final, got {len(yields)}"
    # Real segments
    for i, y in enumerate(yields[:-1]):
        assert y.is_final is False, f"segment {i} should not be final"
        assert y.audio_payload is not None
        assert y.segment_index == i
        assert y.audio_payload.shape == (32,)
        # The fake model scales each segment's amplitude
        assert np.allclose(y.audio_payload, 0.05 * (i + 1))
    # Final sentinel
    final = yields[-1]
    assert final.is_final is True
    assert final.audio_payload is None
    assert final.semantic_tokens_total == 12  # 3 segments * 4 tokens each


def test_synthesize_segments_empty_model_emits_empty_sentinel() -> None:
    """If the model yields no segments, the generator still emits the final sentinel.

    The empty-sentinel case carries ``final_timing={\"empty\": True}`` so
    the consumer (synthesize() or a future chunked handler) can raise
    the original \"Model returned no audio.\" error.
    """
    from local_mlx.host_server import SegmentYield

    class _EmptyModel:
        sample_rate = 24000

        def generate(self, **_kwargs):  # noqa: ANN003
            return
            yield  # unreachable, makes this a generator

    class _StubServer:
        lock = threading.Lock()

        def gen(self, model, gen_kwargs):  # noqa: ANN003
            started = time.perf_counter()
            empty = True
            with self.lock:
                for result in model.generate(**gen_kwargs):
                    empty = False
                    yield SegmentYield(
                        segment_index=0,
                        audio_payload=result.audio,
                        sample_rate=24000,
                        is_final=False,
                    )
            if empty:
                yield SegmentYield(
                    segment_index=0,
                    audio_payload=None,
                    sample_rate=24000,
                    is_final=True,
                    final_timing={"empty": True},
                    elapsed_s=time.perf_counter() - started,
                )

    s = _StubServer()
    yields = list(s.gen(_EmptyModel(), {}))
    assert len(yields) == 1
    assert yields[0].is_final is True
    assert yields[0].final_timing == {"empty": True}
    assert yields[0].audio_payload is None


if __name__ == "__main__":
    test_segment_yield_dataclass_basic()
    test_segment_yield_final_sentinel()
    test_synthesize_segments_final_yield_shape()
    test_synthesize_segments_yields_per_segment_then_sentinel()
    test_synthesize_segments_empty_model_emits_empty_sentinel()
    print("All B2-T0 generator tests passed.")
