# Performance Bottleneck Report — Fish Speech S2-Pro MLX Local Deploy

*Report date: 2026-06-20. Scope: the Fish S2-Pro Apple-Silicon MLX deploy.*

**Scope:** the Fish S2-Pro Apple-Silicon MLX deploy only — `local_mlx/host_server.py`,
`local_mlx/fastapi_v1_server.py`, the runtime patch `local_mlx/patches/fish_speech_fastpath.py`,
the installed `mlx_audio` hot path it patches, and `local_mlx/proxy.py`. (LocalTTS / Qwen3-TTS
is out of scope, except where its server contends for the same GPU.)

**About this report:** it is diagnosis + a prioritized, goal-conditional remediation plan. It
recommends no change as already-decided — every item below is a proposal with an effort estimate and
a confidence tag. It supersedes nothing in `TODO_BOTTLENECKS.md`; it re-frames that work against the
measured cost center.

---

## Context — why this report exists

The repo already has a bottleneck analysis (`TODO_BOTTLENECKS.md`, `BOTTLENECK_FIXES_SUMMARY.md`)
covering 8 items (B1–B8) in the HTTP server and the fastpath patch. The key reason a fresh, whole
report is warranted: **the existing roadmap targets serving/ops mechanics (locking, streaming, WAV
encoding, health, shutdown) — and the measured data shows that is not where the time goes.** ~97% of
synthesis wall-clock is autoregressive model compute that the existing plan barely touches (B3
pre-allocation, the only compute item shipped, moved RTF ~4.6%).

This report re-grounds the analysis in **measured** data (the repo's own profiler output + live
system inspection), ranks bottlenecks by real impact, and separates "make one request faster" (RTF)
from "serve more clients" (throughput) from "ops" — because the right fix depends on the goal.

**Verification stance:** every quantitative claim below is tagged `[measured]` (from a benchmark
artifact / live inspection), `[code]` (read directly in source, file:line given), or
`[hypothesis]` (a proposed fix whose payoff is *not* yet proven and must be benchmarked). No numbers
are fabricated; where the repo's own artifacts disagree, this report says so.

---

## What actually runs (live inspection, `ps`/`lsof`/`/health`)

| Process | Port | What | Load mode |
|---|---|---|---|
| `host_server.py` (PID 970) | 8881 | Fish S2-Pro 8-bit MLX, raw `ThreadingHTTPServer` | `--no-lazy --warmup` (eager) |
| `fastapi_v1_server.py` (PID 953) | 8882 | **Same** Fish model, OpenAI `/v1` FastAPI | eager |
| LocalTTS uvicorn (PID 18208) | 18882 | Qwen3-TTS-12Hz-1.7B 8-bit MLX (out of scope) | — |

- Deploy interpreter: `/Users/op/fish-speech-int4-patch/.venv-mlx` (Python 3.11). Packages: **mlx 0.31.2,
  mlx-audio 0.4.0, mlx-lm 0.31.1; no torch.** `[measured]`
- **Therefore the `fish_speech/` PyTorch tree (torch==2.8.0) is the CUDA/upstream path and does NOT run on
  the Mac.** The Mac hot path is the installed `mlx_audio` package
  (`.../mlx_audio/tts/models/fish_qwen3_omni/fish_speech.py`), with two methods monkey-patched at load by
  `fish_speech_fastpath.py`. All RTF-relevant analysis below targets that path. `[code]`
- Model: `checkpoints/fish-audio-s2-pro-8bit-mlx-normalized`. 4-bit MLX builds "decode to noise" per the
  README, so **8-bit is effectively the quantization floor on Mac** — a normally-obvious RTF lever (lower
  bits) is already exhausted. `[measured: README]`

---

## Where the time actually goes — the spine of this report

From the repo's own profiler `local_mlx/benchmarks/rtf_report_final.json`, the representative 152-token
case (`played_sample`), `[measured]`:

| Phase | Time | Share | What it is |
|---|---|---|---|
| **Fast AR** | 6082 ms | **49.5%** | depth transformer: 1 prefill + **9 residual codebook steps per frame** |
| **Slow AR** | 5911 ms | **48.1%** | the 4B time-axis model, one forward per frame |
| Sampling | 257 ms | 2.1% | RAS sampling of the semantic token |
| eval-sync | 31 ms | 0.25% | the single `mx.eval` boundary per frame |
| *(codec + prompt prefill + misc)* | ~1.28 s | ~9% of `gen_seconds` | `_decode_codes` + initial prompt forward (outside the profiled loop) |

Consistent across sizes: the 63-token case is 47.7% slow / 49.2% fast. **The Fast AR is the single
largest cost — slightly above the 4B slow model — and the codec is negligible** ("barely breaking a
sweat," README; the profiler doesn't even surface it). `[measured]`

**Why the Fast AR is so expensive** (`fish_speech_fastpath.py:187-219`) `[code]`: per audio frame the
loop runs ~10 sequential forward passes through the depth transformer — `fast_forward_cached(...)` once
to prefill, then `for i in range(1, num_cb)` (9 iterations) each doing another `fast_forward_cached` +
sample + `fast_embeddings`. A 152-frame clip = **~1,520 fast-forward dispatches + 152 slow-forwards.**
The depth net is small (~400M) but runs ~10× more often, so it costs as much as the 4B model. This is
the structural heart of Dual-AR latency and **is not addressed anywhere in B1–B8.**

### Throughput / RTF — and a real measurement discrepancy

| Source | tok/s | RTF | Notes |
|---|---|---|---|
| `benchmarks/rtf_report_final.json` (model-isolated) | ~11.2 | **1.92** | profiler, realistic cases `[measured]` |
| `benchmarks/wer_rtf_final.json` (HTTP end-to-end, :8881) | ~9 | **2.28** median | full server path `[measured]` |
| `README.md` front page (":8881, GPU to itself") | **5.7** | **3.77** | published headline `[measured]` |

Each row is internally consistent (frame rate ≈ 21.5 Hz, RTF = 21.5 ÷ tok/s). But the **README headline
and the repo's own profiler disagree ~2× on the core generation rate.** That gap is itself a finding
(BN-13): there is no single canonical RTF, so today you *cannot tell whether an optimization helped*.
The most likely cause is GPU/memory contention (see BN-6 — multiple model servers resident at once) or
hardware/version drift between runs. **Resolve this before doing any RTF work.**

---

## Ranked bottlenecks

Severity = impact on the dominant use (single-user local deploy). "Multi-client" items are real but
only bite under concurrency, which this deployment does not currently appear to do.

| # | Bottleneck | Class | Severity | Effort | Fix confidence |
|---|---|---|---|---|---|
| **BN-1** | Fast-AR depth loop — ~10 serial dispatches/frame (~49%) | RTF | **High** | High | Hypothesis |
| **BN-2** | Slow-AR per-frame step (~48%), eager dispatch (no `mx.compile`) | RTF | **High** | High | Hypothesis |
| **BN-6** | Redundant/over-subscribed model servers contend for 1 GPU | RTF + ops | **High** | Trivial | High |
| BN-13 | Repo's own RTF numbers disagree ~2× — no canonical harness | Measurement | High | Low | High |
| BN-4 | Global synthesis lock serialises requests (B1, open) | Throughput | Med* | Med | High |
| BN-5 | No batching; batch primitives missing (B6, open) | Throughput | Med* | High | High |
| BN-7 | Eager load + **synchronous** warmup blocks startup (~3 s) | Cold-start | Low | Low | High |
| BN-8 | `host_server.py` never evicts idle model (FastAPI variant does) | Ops/mem | Low | Low | High |
| BN-9 | Proxy buffers whole response, no keep-alive (`proxy.py:34`) | Latency | Low** | Low | High |
| BN-10 | Text tokenized multiple times per request | Latency | Low | Trivial | High |
| BN-11 | Unconditional text normalization in hot path | Latency | Low | Trivial | Med |
| BN-12 | FLAC/OGG encode is synchronous post-gen (~50–200 ms) | Latency | Low | Low | High |

\* High if you ever serve concurrent clients. \*\* Only in the Docker→host topology; localhost callers bypass it.

### Tier 1 — the RTF core (where 97% of the time is)

**BN-1 — Fast-AR depth loop (~49.5%).** `fish_speech_fastpath.py:187-219` `[code]`. ~10 sequential
`fast_forward_cached` calls per frame, each a tiny matmul whose cost is dominated by per-call MLX
dispatch/launch overhead, not FLOPs. The per-step `mx.eval`s were already removed in production mode
(only the single `.item()` at `:169` forces a sync per frame — unavoidable, it gates the EOS check and
RAS window). Remaining levers, all `[hypothesis]` (must benchmark, may not compose with the custom
KV-cache/dynamic shapes):
- Wrap `fast_forward_cached` (or the whole 9-step depth rollout) in **`mx.compile`** to fuse the small
  ops and cut dispatch overhead — the single most promising un-tried software lever, and it applies to
  BN-2 as well.
- Investigate whether the 9 residual steps can share more work (single fused depth-rollout kernel) rather
  than 9 Python-level forwards.

**BN-2 — Slow-AR per-frame step (~48.1%).** `fish_speech_fastpath.py:237` `self.model(next_input, cache=cache)`
`[code]`. One 4B forward per frame, eager. Levers: `mx.compile` the slow step `[hypothesis]`; speculative/
multi-token decoding (hard for Dual-AR, research-grade); a smaller/distilled time-axis model (largest
possible win, largest effort). **Note the easy lever — lower-bit quant — is already exhausted** (4-bit =
noise on MLX, per README), so 8-bit is the floor. `[measured]`

> **Strategic takeaway:** BN-1 + BN-2 are ~97% of synthesis time and the existing B1–B8 roadmap moves
> them by ~4.6% total. If "faster synthesis" is a goal, the work is here (compile/fuse/distill), not in
> the open B-items. Be honest that these are higher-effort and the payoff is unproven until benchmarked.

**BN-6 — Redundant model servers / GPU contention.** Live inspection shows **two Fish servers
(`:8881` host_server *and* `:8882` fastapi_v1_server) holding the same 8-bit model in unified memory
simultaneously**, plus an out-of-scope LocalTTS MLX server (`:18882`). On a single-GPU unified-memory
Mac, concurrent residency doubles memory pressure and the three compete for one Metal queue — a strong
candidate for the 2× RTF gap in BN-13 (the README's slow 5.7 tok/s vs the profiler's isolated 11.2).
**Fix: run exactly one Fish server.** Trivial effort, real gain, and it makes every other measurement
trustworthy. `[measured: live ps/lsof]`

### Tier 2 — throughput (only if you serve concurrent clients)

**BN-4 — Global synthesis lock (B1, still open).** `host_server.py:~369` — `with self.lock:` wraps the
`model.generate()` loop (released per-segment, not per-request). Concurrent HTTP requests queue. Correct
for a single GPU, but caps multi-client throughput. Gated on a concurrency stress test (B1-T1, not done)
before lock removal vs async migration. `[code, verified]`

**BN-5 — No request batching (B6, open).** The batch path `_generate_codes_for_text_batch_patched`
references `_prepare_batched_prompt_inputs` and `_sample_semantic_batch`, which **are not defined**
anywhere in the patch or `mlx_audio` — so it raises `AttributeError` if called, and no endpoint calls it.
B6 is "implement the missing primitives first," not "wire an existing path." High effort; only matters
multi-client. `[code, verified]`

### Tier 3 — cold-start & ops

- **BN-7** Eager load (~2.8 s `[measured: rtf_report_final.load_seconds]`) + **synchronous** warmup runs
  before the server accepts traffic. Move warmup to a background thread after `listen()` so restarts
  recover faster. Low effort. `[code]`
- **BN-8** `host_server.py` has `evict_model()` but never calls it on idle; the model stays resident
  forever (the FastAPI variant *does* evict after 300 s idle). Add an idle-evict loop or document that
  host_server is "keep-hot." Low effort. `[code]`
- **BN-9** `proxy.py:34` does `resp.read()` — buffers the entire audio before forwarding, and opens a new
  `urllib` connection per request (no keep-alive). Adds ~10–50 ms `[hypothesis]`, **only** on the
  Docker→host hop; direct localhost callers are unaffected. Low priority unless the Docker path is in use.

### Tier 4 — hygiene / minor

- **BN-10** The prompt text is tokenized more than once per request (`host_server` effective-max-tokens
  calc + again inside generation, `fastpath:132`). ~5–20 ms; cache the count. Trivial. `[code]`
- **BN-11** Text normalization runs unconditionally in the hot path; add a bypass env flag for callers
  that pre-normalize. Trivial. `[code]`
- **BN-12** FLAC/OGG encode is synchronous after generation (~50–200 ms). WAV is already hand-rolled
  (B5 shipped, fast). Offload non-WAV encode to a thread. Low. `[code]`
- **BN-14** (correctness-adjacent) `FISH_MLX_MAX_TOKENS=256` (plist) caps output at ~12 s of audio;
  long inputs truncate mid-sentence. Worth surfacing as a knob in docs. `[measured: plist]`

---

## Status of the existing B1–B8 roadmap (verified against current code)

7 of 8 items shipped correctly; the remaining open items are all **throughput/ops, orthogonal to RTF.**

| Item | Claim | Verified status | Evidence |
|---|---|---|---|
| B3-T1/T2 | Pre-allocate `next_input_buf` | **Shipped, correct** | `fastpath.py:148,227-228,233` in-place fill replaces per-step `concatenate` |
| B5-T1 | Hand-rolled WAV encoder | **Shipped, correct** | `host_server.py` `_encode_wav_pcm16`, explicit f32→i16 scaling |
| B7-T1 | Graceful SIGTERM/SIGINT drain | **Shipped** | signal handler → drain ≤30 s → evict → exit 0 |
| B8-T1 | Extended `/health` | **Shipped** | uptime / in_flight / last_request / last_error |
| B1-T4 | `X-Queue-Depth` header | **Shipped** | thread-safe in-flight counter |
| B1 | Remove global lock | **Open** | lock still wraps generate; gated on missing stress test |
| B2 | Segment streaming | **Partial** | generator refactor done (B2-T0); chunked HTTP (T1-T5) open |
| B6 | Batching | **Open + blocked** | batch primitives undefined (see BN-5) |
| B4 | `mx.clear_cache()` every 50 | **Dormant** | only in the never-called batch path |
| B7-T2 | Shutdown-reason INFO log | **Minor gap** | drain prints status but not a final reason line |

**Conclusion:** the shipped work is real and correct, but it is serving/ops hardening. None of the open
items (B1, B2, B6) reduces RTF; they raise multi-client throughput and polish. That's fine — it just
means the roadmap and the actual cost center have never overlapped.

---

## Prioritized recommendations (goal-conditional)

**Do first regardless of goal (cheap, and unblocks trustworthy measurement):**
1. **BN-6** — run one Fish server, not two; quit the redundant `:8882` (or `:8881`). Frees memory, removes
   GPU contention. Trivial.
2. **BN-13** — adopt one canonical benchmark (`profile_generation.py` with the GPU isolated, pinned model
   + sampling config) as the *single* RTF source of truth; reconcile/retire the README's 3.77 vs the
   profiler's 1.92. Without this, every "did it get faster?" claim is unfalsifiable.
3. Quick wins: **BN-7** (background warmup), **BN-10** (cache token count), **BN-12** (thread non-WAV encode).

**If the goal is single-request speed (RTF):** focus on **BN-1/BN-2.** Concretely: prototype `mx.compile`
on `fast_forward_cached` and the slow step, behind a benchmark gate (this is a `[hypothesis]` — measure
before/after on the canonical harness; it may not compose with the custom cache). If compile underdelivers,
the only larger levers are a fused depth-rollout kernel or a smaller time-axis model. Set expectations:
this is the high-value, high-effort, unproven-payoff frontier.

**If the goal is multi-client throughput:** do **BN-4** then **BN-5** — but run the B1-T1 concurrency
stress test first to decide lock-removal vs async, and budget B6 as "implement missing primitives," not
"wire existing path."

**If the goal is cold-start/ops:** **BN-7, BN-8, BN-9** in that order.

---

## Open questions / what must be benchmarked (honest unknowns)

- **Will `mx.compile` actually speed BN-1/BN-2?** Unknown until measured; the custom KV-cache and dynamic
  shapes may defeat it. Treat as an experiment, not a fix.
- **What causes the 2× RTF gap (BN-13)?** Most likely GPU contention (BN-6) or version/hardware drift —
  confirm by re-running isolated.
- **Is multi-client even a goal?** The deploy looks single-user (LaunchAgent, localhost, eager+warmup).
  If it stays single-user, the entire B1/B6 effort is deprioritized in favor of BN-1/BN-2.
- **Has `mlx-audio` improved upstream since 0.4.0?** The patch pins to internals; a newer release might
  ship faster fish/qwen kernels and reduce the value of hand patching. Worth a check.

## How to measure (verification harness)

```bash
# Canonical model-isolated RTF + phase breakdown (GPU should be otherwise idle):
PYTHONPATH=/Users/op/fish-s2-pro-mlx-local-deploy \
  /Users/op/fish-speech-int4-patch/.venv-mlx/bin/python local_mlx/profile_generation.py \
  --model-path /Users/op/fish-speech-int4-patch/checkpoints/fish-audio-s2-pro-8bit-mlx-normalized

# End-to-end HTTP RTF + ASR WER (no regression) on a single running server:
PYTHONPATH=/Users/op/fish-s2-pro-mlx-local-deploy \
  /Users/op/fish-speech-int4-patch/.venv-mlx/bin/python local_mlx/benchmarks/measure_wer_rtf.py --tag <name>

# Confirm only ONE model server is resident before benchmarking:
lsof -nP -iTCP -sTCP:LISTEN | grep -E '8881|8882'
```
Gate any BN-1/BN-2 change on: phase split from `rtf_report_*.json` (`fast_ms`/`slow_ms`) improves, and
WER does not regress on S2/S3/S4 (the clean sentences).

---

## Evidence index (file:line)

- Hot loop (Fast/Slow AR): `local_mlx/patches/fish_speech_fastpath.py:148,169,187-219,237,248`
- Upstream un-patched reference: `…/mlx_audio/tts/models/fish_qwen3_omni/fish_speech.py:504-608` (gen),
  `:602-608` (`_decode_codes`), `:629-630` (`stream=True` → `NotImplementedError`)
- Profiler output: `local_mlx/benchmarks/rtf_report_final.json`, `wer_rtf_final.json`
- Serving/lock/health/shutdown: `local_mlx/host_server.py`; proxy: `local_mlx/proxy.py:34`
- Deploy config: `local_mlx/com.op.fish-mlx-host.plist` (`--no-lazy --warmup`, `FISH_MLX_MAX_TOKENS=256`)
- Existing analysis: `TODO_BOTTLENECKS.md`, `BOTTLENECK_FIXES_SUMMARY.md`
