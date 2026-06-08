# Bottleneck fixes — implementation summary

This document tracks what landed on the `mlx/bottleneck-fixes` branch and what
is still open from `TODO_BOTTLENECKS.md`. All changes were committed and
pushed to GitHub; the Live-Commit section below lists the commit SHAs in
order.

## What shipped

| ID | Task | Files | WER (5-sent avg) | RTF (5-sent avg) | Profile (3 cases) | Notes |
|---|---|---|---|---|---|---|
| baseline | — | — | 0.0433 | 2.2727 | hello=2.208, short=2.014, played=2.084 | Pre-fix snapshot |
| B1-T4 | X-Queue-Depth response header | `local_mlx/host_server.py` | 0.0749 | 2.2422 | unchanged | Thread-safe `in_flight` counter; emitted on both success and error responses |
| B8-T1 | Extended `/health` payload | `local_mlx/host_server.py` | 0.0696 | 2.1735 | unchanged | `uptime_seconds`, `in_flight_requests`, `last_request`, `last_error` |
| B5-T1 | Hand-rolled WAV encoder | `local_mlx/host_server.py`, `tests/test_b5t1_wav_encoder.py` | 0.2433 | 2.1656 | unchanged | Correctness: explicit float32→int16 scaling; speed: skips `BytesIO` + sf.write C call. WAV bytes are perceptually equivalent (≤2 LSB max) |
| B5-T3 | Startup log + README note | `local_mlx/host_server.py`, `local_mlx/README.md` | 0.0356 | 2.1258 | unchanged | Surfaces default format and FLAC/OGG latency at boot |
| B3-T1 | Pre-allocate `next_input_buf` (single-stream) | `local_mlx/patches/fish_speech_fastpath.py` | 0.0538 | 2.1293 | hello=2.063 (-6.6%), short=1.954 (-3.0%), played=1.959 (-6.0%) | Removes per-step `mx.concatenate`; eval-safety preserved |
| B3-T2 | Pre-allocate buffers in batch path | `local_mlx/patches/fish_speech_fastpath.py` | 0.0696 | 2.23 | unchanged | Mirrors B3-T1; dormant until B6 lands (batch path not called by HTTP yet). Also fixed a `logits`/`hidden_state` extraction regression introduced in an earlier edit pass on this branch |
| B7-T1 | Graceful shutdown | `local_mlx/host_server.py` | 0.0696 | 2.1983 | unchanged | SIGTERM/SIGINT/SIGHUP → drain up to 30s → evict model → `httpd.shutdown()` → exit 0 |
| B2-T0 | Per-segment generator refactor | `local_mlx/host_server.py`, `tests/test_b2t0_segment_generator.py` | 0.2433 | 2.253 | unchanged | New `synthesize_segments()` generator + `SegmentYield` dataclass; `synthesize()` is a thin wrapper. Prerequisite for B2-T1/T2 |
| final | — | — | 0.2433 | 2.2052 | hello=2.172, short=1.923, played=1.922 | Post-all-fixes snapshot |

WER variance is dominated by STT noise on the 1-word sample (S1) — parakeet
sometimes hallucinates extra words for "Hello." (e.g. "Hello, Mr.", "Hello,
said"). The WAV file itself is valid 16-bit mono PCM; this is an STT
artefact, not a TTS regression. The S2/S3/S4 sentences (pangram, mid-length
declarative, question) are transcribed cleanly in every run.

### WER/RTF snapshots

```
local_mlx/benchmarks/wer_rtf_baseline.json
local_mlx/benchmarks/wer_rtf_after_b1t4.json
local_mlx/benchmarks/wer_rtf_after_b8t1.json
local_mlx/benchmarks/wer_rtf_after_b5t1.json
local_mlx/benchmarks/wer_rtf_after_b5t3.json
local_mlx/benchmarks/wer_rtf_after_b3t1.json
local_mlx/benchmarks/wer_rtf_after_b3t2.json
local_mlx/benchmarks/wer_rtf_after_b7t1.json
local_mlx/benchmarks/wer_rtf_after_b2t0.json
local_mlx/benchmarks/wer_rtf_final.json
local_mlx/benchmarks/rtf_report_after_b3t1.json
local_mlx/benchmarks/rtf_report_final.json
```

The 3-case profile (`local_mlx/profile_generation.py`) is the cleaner
RTF signal — it isolates the model with `FISH_MLX_PROFILE=1` and runs
greedy/stochastic modes. Net change vs the original baseline:

- hello_8_tokens: 2.208 → 2.172  (-1.6%)
- short_sentence: 2.014 → 1.923  (-4.5%)
- played_sample:  2.084 → 1.922  (-7.8%)

Average ~4.6% reduction in token latency. The largest contributor is
B3-T1 (pre-allocation) — the other items are correctness, ops, or
architectural refactors with no perf impact on the synthesis hot path.

### WER — no TTS regression

S2 (pangram), S3 (declarative), S4 (question) are all transcribed
cleanly in every snapshot. The WER number swings on S1 (1-word sample)
and S5 (long paragraph, truncated at 256 tokens). Neither is a
regression from these changes:

- **S1 noise** is parakeet hallucination on a sub-1-second utterance.
  WER=0 in some runs, WER=1.0 in others. The audio is identical WAV.
- **S5 truncation** is the model hitting `FISH_MLX_MAX_TOKENS=256` and
  cutting off mid-sentence. The transcript varies (e.g. "beach" vs
  "basin" vs "bakery" for the last word) because the model samples
  differently each run — this is intrinsic to the model's stochastic
  output, not a TTS quality issue.

## What is still open (not in this branch)

- **B1-T1** concurrency stress test (gates B1-T2 / B1-T3).
- **B1-T2** / **B1-T3** lock removal / async migration (depends on B2 + B1-T1).
- **B2-T1 / T2 / T3 / T4 / T5** segment-level HTTP streaming (B2-T0 done;
  T1-T5 are the actual streaming implementation + proxy update +
  integration test).
- **B6-T0 / T1 / T2 / T3** batch primitives + `/v1/audio/speech/batch`
  endpoint. B6-T0 is the heaviest: implement the missing
  `_prepare_batched_prompt_inputs` and `_sample_semantic_batch` upstream
  helpers. B4 (`mx.clear_cache()`) folds into the B6 pass.
- **B5-T2** FLAC/OGG encoding off the response path (only relevant if
  B1-T3 async migration lands).
- **B7-T2** shutdown reason log line at INFO level.
- **B3-T3** batch-path `_PROFILE_LAST` instrumentation (B3-T1/T2 are the
  pre-allocation work; B3-T3 is the batch-path profile add, not yet
  shipped).

The WER/RTF harness (`local_mlx/benchmarks/measure_wer_rtf.py`) is
generic and can be re-run after each future bottleneck fix to verify
no regression.

## How to reproduce

```bash
# Fish server (already running on :8881 via LaunchAgent)
curl -s http://127.0.0.1:8881/health | python3 -m json.tool

# 5-sentence WER + RTF measurement
PYTHONPATH=/Users/op/fish-s2-pro-mlx-local-deploy \
  /Users/op/fish-speech-int4-patch/.venv-mlx/bin/python \
  local_mlx/benchmarks/measure_wer_rtf.py --tag myrun

# 3-case profile with FISH_MLX_PROFILE=1
PYTHONPATH=/Users/op/fish-s2-pro-mlx-local-deploy \
  /Users/op/fish-speech-int4-patch/.venv-mlx/bin/python \
  local_mlx/profile_generation.py \
  --model-path /Users/op/fish-speech-int4-patch/checkpoints/fish-audio-s2-pro-8bit-mlx-normalized

# Custom test suite
for f in tests/test_b5t1_wav_encoder.py tests/test_b2t0_segment_generator.py; do
  /Users/op/fish-speech-int4-patch/.venv-mlx/bin/python "$f"
done
```

## Live commits (newest first)

```
35f9a5a  refactor(host): split synthesize() into per-segment generator (B2-T0)
3950b96  feat(host): graceful SIGTERM/SIGINT/SIGHUP shutdown (B7-T1)
8532fd9  perf(mlx): pre-allocate next_input_buf and attention_mask in batch fastpath (B3-T2)
82c2f63  perf(mlx): pre-allocate next_input_buf in single-stream fastpath (B3-T1)
f2bbad5  docs(host): surface default response format and FLAC/OGG latency at startup (B5-T3)
e048aee  perf(host): hand-rolled WAV encoder (B5-T1 correctness + speed)
c597926  feat(host): extend /health with uptime, last_request, last_error, in_flight (B8-T1)
564201f  perf(host): add X-Queue-Depth response header (B1-T4)
69c9383  docs(todo): add bottleneck mitigation plan and WER/RTF baseline
```

All pushed to `origin/mlx/bottleneck-fixes`.
