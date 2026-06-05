#!/usr/bin/env python3
"""Benchmark Fish S2 MLX generation RTF against plan acceptance gates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("FISH_MLX_PROFILE", "1")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mlx.core as mx
from mlx_audio.tts.utils import load_model

from local_mlx.patches.fish_speech_fastpath import apply_fish_speech_patch, get_profile_stats

BENCHMARK_CASES = [
    {
        "name": "hello_8_tokens",
        "text": "Hello.",
        "max_tokens": 8,
        "rtf_threshold": 3.0,
        "baseline_rtf": 3.99,
    },
    {
        "name": "short_sentence",
        "text": "The quick brown fox jumps over the lazy dog.",
        "max_tokens": 256,
        "rtf_threshold": 1.6,
        "baseline_rtf": 2.07,
    },
    {
        "name": "played_sample",
        "text": (
            "This is a slightly longer sample used to validate Fish S2 MLX latency "
            "after the fastpath patch."
        ),
        "max_tokens": 256,
        "rtf_threshold": 1.5,
        "baseline_rtf": 2.08,
    },
]


def run_case(model, case: dict, *, greedy: bool) -> dict:
    temperature = 0.0 if greedy else 0.7
    started = time.perf_counter()
    audio_samples = 0
    semantic_tokens = 0
    sample_rate = getattr(model, "sample_rate", 24000)

    for result in model.generate(
        text=case["text"],
        max_tokens=case["max_tokens"],
        temperature=temperature,
        verbose=False,
        stream=False,
    ):
        audio_samples += int(getattr(result, "samples", 0) or 0)
        semantic_tokens += int(getattr(result, "token_count", 0) or 0)
        sample_rate = int(getattr(result, "sample_rate", sample_rate) or sample_rate)

    mx.eval(mx.array(0))
    gen_seconds = max(time.perf_counter() - started, 1e-6)
    audio_seconds = audio_samples / max(sample_rate, 1)
    rtf = gen_seconds / max(audio_seconds, 1e-6)
    passed = rtf < case["rtf_threshold"]

    return {
        "name": case["name"],
        "text": case["text"],
        "max_tokens": case["max_tokens"],
        "greedy": greedy,
        "gen_seconds": round(gen_seconds, 4),
        "audio_seconds": round(audio_seconds, 4),
        "semantic_tokens": semantic_tokens,
        "rtf": round(rtf, 4),
        "baseline_rtf": case["baseline_rtf"],
        "rtf_threshold": case["rtf_threshold"],
        "passed": passed,
        "patch_profile": get_profile_stats(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="checkpoints/fish-audio-s2-pro-8bit-mlx-normalized",
    )
    parser.add_argument(
        "--output",
        default="local_mlx/benchmarks/rtf_report.json",
    )
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")

    apply_fish_speech_patch()
    print(f"Loading model from {model_path}...", flush=True)
    load_started = time.perf_counter()
    model = load_model(model_path, lazy=False, strict=True)
    load_seconds = time.perf_counter() - load_started
    mx.eval(mx.array(0))

    print("Warmup synthesis...", flush=True)
    for _ in model.generate(text="Hi.", max_tokens=8, temperature=0.7, verbose=False):
        pass
    mx.eval(mx.array(0))

    results = {
        "model_path": str(model_path),
        "load_seconds": round(load_seconds, 4),
        "greedy": args.greedy,
        "cases": [],
        "all_passed": True,
    }

    for case in BENCHMARK_CASES:
        print(f"Running case: {case['name']}...", flush=True)
        outcome = run_case(model, case, greedy=args.greedy)
        results["cases"].append(outcome)
        if not outcome["passed"]:
            results["all_passed"] = False
        print(
            f"  RTF={outcome['rtf']:.3f} "
            f"(threshold<{outcome['rtf_threshold']}, baseline={outcome['baseline_rtf']}) "
            f"{'PASS' if outcome['passed'] else 'FAIL'}",
            flush=True,
        )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote benchmark report to {output_path}", flush=True)
    print(f"All cases passed: {results['all_passed']}", flush=True)

    if not results["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
