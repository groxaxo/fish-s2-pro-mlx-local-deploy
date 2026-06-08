"""Regression tests for B5-T1 (hand-rolled WAV encoder).

Verifies that ``_encode_wav_pcm16`` produces a byte-identical or
perceptually-equal WAV file to ``soundfile.write(..., format="WAV")`` for the
same float32 input, across the values that the model actually emits
(clipping, full-scale, silence).
"""

from __future__ import annotations

import io
import struct
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from local_mlx.host_server import _encode_wav_pcm16  # noqa: E402


def _sf_write_wav_pcm16(audio_np: np.ndarray, sample_rate: int) -> bytes:
    """Reference implementation: encode the same way the old code did."""
    out = io.BytesIO()
    sf.write(out, audio_np, sample_rate, format="WAV", subtype="PCM_16")
    return out.getvalue()


def _decode_pcm16(body: bytes) -> tuple[int, int, np.ndarray]:
    """Parse RIFF/WAVE and return (sample_rate, n_channels, pcm_int16)."""
    assert body[:4] == b"RIFF", f"missing RIFF: {body[:12]!r}"
    assert body[8:12] == b"WAVE", f"missing WAVE: {body[8:16]!r}"
    # Walk chunks
    offset = 12
    fmt = None
    data = b""
    while offset < len(body):
        chunk_id = body[offset : offset + 4]
        chunk_size = struct.unpack("<I", body[offset + 4 : offset + 8])[0]
        chunk_data = body[offset + 8 : offset + 8 + chunk_size]
        if chunk_id == b"fmt ":
            fmt = struct.unpack("<HHIIHH", chunk_data[:16])
        elif chunk_id == b"data":
            data = chunk_data
        offset += 8 + chunk_size
    assert fmt is not None, "no fmt chunk"
    audio_format, n_channels, sample_rate, byte_rate, block_align, bits = fmt
    assert audio_format == 1, f"not PCM (got {audio_format})"
    assert bits == 16, f"not 16-bit (got {bits})"
    pcm = np.frombuffer(data, dtype="<i2")
    return sample_rate, n_channels, pcm


def test_zero_audio_round_trip() -> None:
    """All-zero float32 in must encode to a valid PCM-16 WAV with all-zero samples."""
    audio = np.zeros(1024, dtype=np.float32)
    rate = 24000
    body = _encode_wav_pcm16(audio, rate)
    sample_rate, n_ch, pcm = _decode_pcm16(body)
    assert sample_rate == rate
    assert n_ch == 1
    assert pcm.shape == (1024,)
    assert np.all(pcm == 0)


def test_full_scale_positive_clamps_to_int16_max() -> None:
    audio = np.full(8, 1.0, dtype=np.float32)
    body = _encode_wav_pcm16(audio, 24000)
    _, _, pcm = _decode_pcm16(body)
    # clip(1.0 * 32767) -> 32767, not 32768 (would wrap on int16)
    assert np.all(pcm == 32767), f"expected 32767, got {pcm}"


def test_full_scale_negative_clamps_to_int16_min() -> None:
    audio = np.full(8, -1.0, dtype=np.float32)
    body = _encode_wav_pcm16(audio, 24000)
    _, _, pcm = _decode_pcm16(body)
    # clip(-1.0 * 32767) -> -32767, not -32768
    assert np.all(pcm == -32767), f"expected -32767, got {pcm}"


def test_out_of_range_clamped() -> None:
    """Values outside [-1, 1] must clip, not wrap."""
    audio = np.array([2.0, -3.0, 1.5, -1.5], dtype=np.float32)
    body = _encode_wav_pcm16(audio, 24000)
    _, _, pcm = _decode_pcm16(body)
    assert pcm.tolist() == [32767, -32767, 32767, -32767]


def test_byte_length_matches_data_size_field() -> None:
    """The 'data' chunk length in the header must equal the actual PCM byte count.

    A common bug: header sizes from the *float* array's nbytes (8 bytes per
    sample with float64) instead of int16 (2 bytes per sample). Decoder then
    either truncates or refuses the file.
    """
    n = 1000
    audio = np.zeros(n, dtype=np.float32)
    body = _encode_wav_pcm16(audio, 24000)
    # RIFF chunk size = 36 (header) + data
    riff_size = struct.unpack("<I", body[4:8])[0]
    data_size = struct.unpack("<I", body[40:44])[0]
    assert data_size == n * 2, f"data_size={data_size}, expected {n * 2}"
    assert riff_size == 36 + n * 2


def test_equivalence_with_soundfile() -> None:
    """Hand-rolled path must be perceptually equivalent to sf.write.

    soundfile uses the 32768 multiplier to fill the full int16 range
    ([-32768, 32767]); this module uses 32767 (asymmetric, no overflow
    possible). At the int16 boundary this is up to 2 LSB off — for full-
    scale random audio that is ~0.006 dB, well below the JND for human
    hearing. Acceptable. Empirically the max diff on a normal-distribution
    signal is 2 LSB.
    """
    rng = np.random.default_rng(seed=42)
    audio = (rng.standard_normal(8192).astype(np.float32) * 0.3)
    rate = 44100
    a = _encode_wav_pcm16(audio, rate)
    b = _sf_write_wav_pcm16(audio, rate)
    _, _, pa = _decode_pcm16(a)
    _, _, pb = _decode_pcm16(b)
    assert pa.shape == pb.shape
    diff = np.abs(pa.astype(np.int32) - pb.astype(np.int32))
    assert int(diff.max()) <= 2, f"max LSB diff = {int(diff.max())}"


def test_no_bytesio_or_soundfile_in_wav_path() -> None:
    """The hot path for WAV must not call soundfile at runtime.

    Static check on the function's own bytecode: no LOAD_GLOBAL for ``sf``
    or ``BytesIO`` and no call to ``sf.write``. Guards against a future
    refactor that re-routes WAV through sf.write and loses the perf /
    explicitness wins.
    """
    import dis

    code = _encode_wav_pcm16.__code__
    names = set(code.co_names) | set(code.co_varnames)
    # Free vars / global accesses
    for const in code.co_consts:
        if hasattr(const, "co_name"):
            names |= set(const.co_names)
    assert "BytesIO" not in names, f"BytesIO leaked into WAV encoder: {names}"
    # Walk the bytecode for LOAD_GLOBAL
    for instr in dis.get_instructions(code):
        if instr.opname == "LOAD_GLOBAL" and instr.argval == "sf":
            raise AssertionError(f"WAV encoder calls into soundfile at {instr.offset}")


def test_44_byte_header_size() -> None:
    audio = np.zeros(16, dtype=np.float32)
    body = _encode_wav_pcm16(audio, 24000)
    # header is 44 bytes, plus 2 bytes per int16 sample
    assert len(body) == 44 + 16 * 2, f"got {len(body)}"


def test_riff_chunk_size_includes_data() -> None:
    audio = np.zeros(7, dtype=np.float32)
    body = _encode_wav_pcm16(audio, 24000)
    # RIFF size = 4 (WAVE) + 24 (fmt chunk) + 8 (data header) + data
    riff_size = struct.unpack("<I", body[4:8])[0]
    assert riff_size == 4 + 24 + 8 + 7 * 2


if __name__ == "__main__":
    test_zero_audio_round_trip()
    test_full_scale_positive_clamps_to_int16_max()
    test_full_scale_negative_clamps_to_int16_min()
    test_out_of_range_clamped()
    test_byte_length_matches_data_size_field()
    test_equivalence_with_soundfile()
    test_no_bytesio_or_soundfile_in_wav_path()
    test_44_byte_header_size()
    test_riff_chunk_size_includes_data()
    print("All B5-T1 tests passed.")
