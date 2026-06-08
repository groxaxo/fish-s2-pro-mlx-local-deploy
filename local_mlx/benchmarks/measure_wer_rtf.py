"""
Baseline 5-sentence measurement: RTF + WER for Fish S2 MLX.

Generates 5 fixed sentences (varied length, varied content), transcribes each
back through the local STT (Parakeet CoreML) at /v1/audio/transcriptions on
port 5093, and writes a JSON report to local_mlx/benchmarks/wer_rtf_baseline.json
(or _<tag>.json when --tag is given).

Usage:
  python local_mlx/benchmarks/measure_wer_rtf.py \
    --tts http://127.0.0.1:8881 \
    --stt http://127.0.0.1:5093 \
    --tag baseline
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

# Five fixed sentences chosen to exercise the model:
# - S1: trivial (1 word, baseline TTFB)
# - S2: pangram (covers most English phonemes)
# - S3: short declarative (typical short-form TTS use)
# - S4: question with punctuation (model has to handle `?`)
# - S5: longer paragraph (~120 tokens, exercises max_tokens cap and a/c mixing)
SENTENCES: list[dict[str, str]] = [
    {"id": "S1", "text": "Hello."},
    {
        "id": "S2",
        "text": "The quick brown fox jumps over the lazy dog, then rests beneath a silver birch.",
    },
    {
        "id": "S3",
        "text": "Speech synthesis is computationally expensive, but modern accelerators help.",
    },
    {
        "id": "S4",
        "text": "Can you hear the difference between warm and cold audio quality?",
    },
    {
        "id": "S5",
        "text": (
            "On a clear morning, the harbour glittered as fishing boats returned "
            "with the night's catch. Vendors shouted prices, gulls wheeled overhead, "
            "and the air smelled of salt, smoke, and fresh bread from the bakery on the quay."
        ),
    },
]


def _wer(reference: str, hypothesis: str) -> float:
    """Word-level Levenshtein distance / reference length (matches tools/correlate_with_whisper.py)."""
    r = reference.strip().split()
    h = hypothesis.strip().split()
    n = len(r)
    if n == 0:
        return 0.0 if len(h) == 0 else 1.0
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(r)][len(h)] / float(n)


def _normalize_for_wer(text: str) -> str:
    """Normalise text so transcription casing/punctuation noise doesn't inflate WER."""
    import re
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


@dataclass
class SentenceResult:
    id: str
    text: str
    status: str
    tts_elapsed_s: float | None = None
    audio_seconds: float | None = None
    rtf: float | None = None
    semantic_tokens: int | None = None
    transcript: str | None = None
    wer: float | None = None
    error: str | None = None


def _measure_one(tts_url: str, stt_url: str, sentence: dict[str, str], timeout_s: int) -> SentenceResult:
    result = SentenceResult(id=sentence["id"], text=sentence["text"], status="ok")
    payload = {
        "input": sentence["text"],
        "model": "fish-audio-s2-pro-8bit-mlx",
        "voice": "default",
        "response_format": "wav",
    }
    try:
        t0 = time.perf_counter()
        r = requests.post(tts_url, json=payload, timeout=timeout_s)
        tts_elapsed = time.perf_counter() - t0
        r.raise_for_status()
    except Exception as exc:
        result.status = "tts_error"
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    result.tts_elapsed_s = round(tts_elapsed, 4)
    try:
        result.rtf = float(r.headers.get("X-RTF", "0") or 0)
        result.audio_seconds = float(r.headers.get("X-Audio-Seconds", "0") or 0)
        result.semantic_tokens = int(r.headers.get("X-Semantic-Tokens", "0") or 0)
    except (TypeError, ValueError):
        pass

    audio_bytes = r.content
    if not audio_bytes:
        result.status = "empty_audio"
        result.error = "TTS returned 0 bytes"
        return result

    # Write to temp .wav (extension matters for the speech-server's MIME detection)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        files = {"file": (os.path.basename(tmp_path), open(tmp_path, "rb"), "audio/wav")}
        tr = requests.post(
            f"{stt_url.rstrip('/')}/v1/audio/transcriptions",
            files=files,
            data={"response_format": "json"},
            timeout=timeout_s,
        )
        files["file"][1].close()
        tr.raise_for_status()
        transcript = tr.json().get("text", "")
    except Exception as exc:
        result.status = "stt_error"
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    result.transcript = transcript
    result.wer = round(_wer(_normalize_for_wer(sentence["text"]), _normalize_for_wer(transcript)), 4)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts", default="http://127.0.0.1:8881/v1/audio/speech")
    parser.add_argument("--stt", default="http://127.0.0.1:5093")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"=== WER + RTF measurement: {args.tag} ===")
    print(f"TTS: {args.tts}")
    print(f"STT: {args.stt}")
    print()

    # Warmup: 1 request to make sure the model is hot before timing
    print("Warmup...")
    try:
        requests.post(
            args.tts,
            json={"input": "warmup", "model": "fish-audio-s2-pro-8bit-mlx", "voice": "default", "response_format": "wav"},
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"  warmup failed (continuing): {exc}")

    results: list[dict[str, Any]] = []
    for sent in SENTENCES:
        print(f"[{sent['id']}] {sent['text'][:60]}{'...' if len(sent['text']) > 60 else ''}")
        r = _measure_one(args.tts, args.stt, sent, args.timeout)
        results.append(r.__dict__)
        if r.status == "ok":
            print(
                f"  RTF={r.rtf:.3f}  audio={r.audio_seconds:.2f}s  gen={r.tts_elapsed_s:.2f}s  "
                f"tokens={r.semantic_tokens}  WER={r.wer:.4f}"
            )
            print(f"  transcript: {r.transcript!r}")
        else:
            print(f"  ERROR ({r.status}): {r.error}")
        print()

    successful = [r for r in results if r["status"] == "ok"]
    wer_values = [r["wer"] for r in successful if r["wer"] is not None]
    rtf_values = [r["rtf"] for r in successful if r["rtf"] is not None]
    elapsed_values = [r["tts_elapsed_s"] for r in successful if r["tts_elapsed_s"] is not None]
    audio_values = [r["audio_seconds"] for r in successful if r["audio_seconds"] is not None]

    summary = {
        "tag": args.tag,
        "tts_url": args.tts,
        "stt_url": args.stt,
        "n_sentences": len(SENTENCES),
        "n_successful": len(successful),
        "avg_wer": round(statistics.mean(wer_values), 4) if wer_values else None,
        "max_wer": round(max(wer_values), 4) if wer_values else None,
        "avg_rtf": round(statistics.mean(rtf_values), 4) if rtf_values else None,
        "median_rtf": round(statistics.median(rtf_values), 4) if rtf_values else None,
        "avg_gen_seconds": round(statistics.mean(elapsed_values), 4) if elapsed_values else None,
        "avg_audio_seconds": round(statistics.mean(audio_values), 4) if audio_values else None,
        "sentences": results,
    }

    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent / f"wer_rtf_{args.tag}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote report to {output_path}")
    print(
        f"Summary: avg_wer={summary['avg_wer']}  avg_rtf={summary['avg_rtf']}  "
        f"avg_gen={summary['avg_gen_seconds']}s  avg_audio={summary['avg_audio_seconds']}s  "
        f"({summary['n_successful']}/{summary['n_sentences']} ok)"
    )
    return 0 if summary["n_successful"] == summary["n_sentences"] else 1


if __name__ == "__main__":
    sys.exit(main())
