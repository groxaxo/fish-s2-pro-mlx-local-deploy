# Bottleneck Mitigation Plan — Fish S2 Pro MLX Local Deploy

Identified from code audit of `local_mlx/host_server.py` and
`local_mlx/patches/fish_speech_fastpath.py`.

---

## Bottleneck 1 — Global synthesis lock serialises all requests

**File:** `local_mlx/host_server.py:144`
**Impact:** High — `ThreadingHTTPServer` spins threads per connection, but
`with self.lock` wraps the entire `model.generate()` call. Concurrent HTTP
requests queue and do not actually run in parallel.

**Two-part question:** (a) Does MLX graph evaluation tolerate concurrent
reads of the same model weights? Yes — MLX on Apple Silicon uses separate
Metal command buffers per stream; concurrent forward passes are safe.
(b) Is `mlx_audio.model.generate()` reentrant? The upstream method is a
Python generator — each call creates a fresh generator with local state
(no shared instance-level mutations observed in the single-stream path).
But the upstream may mutate `self`-level caches or tokeniser state in
future versions. The full answer requires a stress test.

### Tasks

- [ ] **B1-T1** Write a concurrency stress test script
  (`tools/mlx_concurrency_stress.py`). Launch N concurrent
  `model.generate()` calls on the same model instance with no lock. Assert:
  no GPU OOM, no NaN outputs, no Python-level exceptions, per-request RTF
  degradation < 30% versus sequential baseline. Output a pass/fail report.
- [ ] **B1-T2** If T1 passes: remove `with self.lock` from `synthesize()`
  at `host_server.py:144`, keeping the double-checked lock only in
  `ensure_model()` (line 66) for thread-safe lazy loading.
- [ ] **B1-T3** If T1 fails or GPU contention exceeds 30%: migrate from
  `ThreadingHTTPServer` to an asyncio server (`aiohttp` or `uvicorn`) with
  a single synthesis worker thread fed by an `asyncio.Queue`. **Constraint:**
  this migration must happen *after* B2 (streaming) ships on the current
  server. Do not perform B1-T3 before B2 is stable.
- [ ] **B1-T4** Add an `X-Queue-Depth` response header on every response.
  Track depth with a plain `int` guarded by a dedicated `threading.Lock`
  (there is **no** `threading.atomic` in the stdlib — read-modify-write on a
  bare int is not GIL-safe across `+=`). Increment on `do_POST` entry,
  decrement in a `finally` so it survives exceptions. Store the counter on the
  `FishMLXServer` instance (shared across handler threads), not on the
  per-request `Handler`. Callers back-pressure when depth is high.

---

## Bottleneck 2 — No streaming: full audio buffered before first byte

**File:** `local_mlx/host_server.py:133` (`"stream": False`),
`host_server.py:140–148` (chunk accumulation loop)
**Impact:** Medium — caller must wait for the entire synthesis to finish
before receiving any audio. Real benefit is bounded by segment granularity
(see below): only multi-segment inputs gain TTFB.

> ⚠️ **GROUND TRUTH (verified against the deployed
> `/Users/op/fish-speech-int4-patch/.venv-mlx/.../fish_qwen3_omni/fish_speech.py`).
> The original plan's streaming model was wrong on two counts — do not follow it:**
>
> 1. **`stream=True` is NOT supported.** `Model.generate()` raises
>    `NotImplementedError("Fish Speech streaming is not implemented yet.")`
>    at `fish_speech.py:629–630`. Flipping the flag (old B2-T1) crashes every
>    request. **Do not set `stream=True`.**
> 2. **`generate()` does NOT yield per-frame.** It yields **one
>    `GenerationResult` per text *segment*** (`fish_speech.py:661–720`).
>    Segments come from `split_text_by_speaker()` +
>    `group_turns_into_batches(..., max_bytes=chunk_length)` with
>    `chunk_length` defaulting to **200 bytes** (`generate()` signature line
>    623). Each segment's *entire* audio is codec-decoded in one shot by
>    `_decode_codes()` (`fish_speech.py:602–608`) **before** it is yielded.
>
> **Consequence:** the only streaming achievable without new model code is
> **segment-level** — emit each yielded segment's PCM as an HTTP chunk as it
> arrives. A short single utterance is **one** segment → one yield → **zero
> TTFB improvement**. TTFB only improves for inputs that span multiple
> segments (text longer than ~200 bytes, or multiple speakers). True
> per-frame/per-token streaming would require implementing incremental codec
> decode inside the fastpath patch (decode partial code windows mid-loop) —
> that is a separate, large effort tracked in **Out of scope**, not B2.

### Tasks

- [ ] **B2-T0** (Prerequisite — architectural) Today `synthesize()`
  (`host_server.py:99`) fully buffers: it accumulates `chunks`, encodes once
  with `sf.write()`, and **returns** `(bytes, media_type, timing)`, which
  `do_POST` (line 260) hands to `_bytes()` (line 189). Chunked streaming
  (B2-T2) cannot live in the handler until `synthesize()` is refactored to a
  *generator* (or a callback) that yields PCM per segment as the model
  produces it, with timing reported via a trailer/side-channel rather than a
  return value. Decide the contract here before B2-T2/T4. **Lock caveat:** the
  generation loop currently runs inside `with self.lock` (line 144). If the
  same lock wraps a streaming generator, it is held across socket writes to
  the client — a slow consumer then blocks all other synthesis. Plan to
  release the lock between segments (lock the model step, not the socket
  write), or accept serialized streaming until B1 lands. Note this dependency
  on B1.
- [ ] **B2-T1** Keep `stream=False` (the only working mode). Stream at the
  **segment** level: the existing `for result in model.generate(**gen_kwargs)`
  loop (`host_server.py:145`) already iterates one `GenerationResult` per
  segment. Instead of appending to `chunks` and encoding once at the end,
  hand each `result.audio` to the handler as it is produced. No change to
  `gen_kwargs["stream"]`. Optionally expose `chunk_length` as a request/env
  knob so callers can trade segment size for finer TTFB granularity (smaller
  `chunk_length` → more segments → earlier first byte, at some quality/seam
  cost — benchmark before lowering the 200-byte default).
- [ ] **B2-T2** Switch the HTTP response to chunked transfer encoding. Write
  one HTTP chunk per generated **segment** (not per frame):
  1. Call `self.send_response(HTTPStatus.OK)`
  2. Send `Transfer-Encoding: chunked` header
  3. Omit `Content-Length`
  4. Call `self.end_headers()`
  5. For each segment's PCM bytes: `self.wfile.write(f"{len(chunk):x}\r\n".encode()); self.wfile.write(chunk); self.wfile.write(b"\r\n")`
  6. After the last segment: `self.wfile.write(b"0\r\n\r\n")`
  Flush after each segment so the byte actually leaves the process.
- [ ] **B2-T3** Streaming audio format strategy (segment-level):
  - **WAV (default):** Send raw PCM (int16, scaled per B5-T1 — model audio is
    `float32` in [-1, 1]) with sample rate, bit depth, and channel count in
    response headers (`X-Audio-Sample-Rate`, `X-Audio-Bit-Depth`,
    `X-Audio-Channels`). Client reconstructs WAV locally. *Do not* attempt to
    patch a WAV header on close — chunked HTTP response headers are fixed
    after `send_response()`.
  - **OGG:** `soundfile` writes self-contained OGG pages per `sf.write()`
    call; encoding each segment independently produces concatenable streams
    only if the client treats them as separate logical streams — verify with a
    streaming OGG test or fall back to non-streamed OGG.
  - **FLAC:** `soundfile` FLAC encoder buffers the full stream before
    writing. Document FLAC as **not streamable**: for `response_format=flac`,
    fall back to the current buffered (non-chunked) response path.
- [ ] **B2-T4** Update `local_mlx/proxy.py:27–40` to forward chunked
  responses without buffering. Replace `urllib.request.urlopen` (which
  buffers via `resp.read()` at line 35) with `http.client.HTTPConnection`
  that reads upstream chunks and re-emits them via chunked encoding. Propagate
  `X-Audio-Sample-Rate` and related audio headers. (The proxy is the Docker→
  macOS hop; without this change, chunked host responses get re-buffered here
  and TTFB gains are lost end-to-end.)
- [ ] **B2-T5** Add integration test `tests/test_streaming_ttfb.py`. **The
  input must span multiple segments** or the test proves nothing (a short
  utterance is a single segment with no TTFB gain):
  1. POST to `/v1/audio/speech` with a **multi-segment** input — e.g. several
     sentences totalling > 400 bytes, or multi-speaker text — so
     `group_turns_into_batches` yields ≥ 2 segments.
  2. Measure TTFB (time to first HTTP chunk received) vs. total completion.
  3. Assert TTFB is meaningfully less than total (e.g. `ttfb < 0.6 * total`),
     **not** an absolute wall-clock bound (RTF varies by host).
  4. Add a second case: a short single-segment input, asserting it still
     returns correct, decodable audio (streaming path must not regress the
     one-segment case even though it shows no TTFB win).
  5. Assert produced audio is decodable for `response_format: "wav"`; verify
     `"ogg"`/`"flac"` per the B2-T3 fallbacks.

---

## Bottleneck 3 — mx.concatenate inside the slow-model loop

**File:** `local_mlx/patches/fish_speech_fastpath.py:214` (single-stream),
`:386` (batch)
**Impact:** Medium — each semantic step builds `next_input` via
`mx.concatenate`, forcing MLX to materialise both operands and allocate a
new buffer every step. For a 256-token sequence that is 256 small
allocations + copy barriers.

**Note:** The `codebook_buf` concatenation (the original bottleneck) was
already pre-allocated at lines 187 and 354. This bottleneck targets only
the remaining `next_input` concatenation and the `attention_mask` growth in
the batch path.

### Tasks

- [ ] **B3-T1** Single-stream path (`:214`): Pre-allocate `next_input_buf`
  of shape `(1, num_cb + 1, 1)` before the step loop. At each step:
  ```python
  next_input_buf[:, 0, 0] = semantic_token
  next_input_buf[:, 1:, 0] = codebook_row  # codebook_row shape (num_cb,) broadcast
  ```
  Pass `next_input_buf` directly to `self.model(...)` — eliminates the
  per-step `mx.concatenate` at line 214. **Lazy-eval caveat:** reusing one
  buffer across iterations is only correct because the production path already
  forces `mx.eval(next_input)` each step (line 239 / profile-path lines
  224, 239), materialising the value before the next overwrite. Keep that eval
  (or eval the buffer explicitly) — dropping it lets a later step's write
  alias an earlier step's still-unevaluated graph and corrupt output. Same
  caveat applies to the batch `next_input_buf` in B3-T2 (eval at line 394).
- [ ] **B3-T2** Batch path (`:386–392`):
  1. Pre-allocate `next_input_buf` of shape `(batch_size, num_cb + 1, 1)`
     before the outer step loop. In-place assign `semantic_token[:, None]`
     to column 0 and `codebook_buf` to columns 1: at each step.
  2. Pre-allocate `attention_mask` at the maximum sequence length
     (`max_budget + initial_prompt_len`) and track `mask_len`. On each model
     call, slice `attention_mask[:, :mask_len]` instead of growing via
     `mx.concatenate` at line 389.
- [ ] **B3-T3** Profile before/after:
  1. Add `_PROFILE_LAST` instrumentation to the batch path
     (`_generate_codes_for_text_batch_patched:260+`) — currently only the
     single-stream path records profile stats (lines 246–255). This is a
     ~10-line addition that gates `FISH_MLX_PROFILE=1` in the batch path.
  2. Run `python local_mlx/profile_generation.py --model-path <path>` before
     and after changes.
  3. Gate: `total_ms` reduction ≥ 5%.

---

## Bottleneck 4 — mx.clear_cache() every 50 steps in batch path

**File:** `local_mlx/patches/fish_speech_fastpath.py:406–407`
**Impact:** None currently — the batch path
(`_generate_codes_for_text_batch_patched`) is not called by the current
HTTP server (only `_generate_codes_for_batch_patched` is, via
`model.generate()`). Becomes **Medium** after B6 (batching) ships.

`mx.clear_cache()` forces the MLX command buffer to flush and frees
intermediate tensor cache. On Apple Silicon this stalls the ANE/GPU
pipeline. At 50-step intervals on a 256-token sequence it fires 5 times
per batch call.

### Tasks

- [ ] **B4-T1** Benchmark memory pressure without `clear_cache()` on typical
  batch inputs (batch_size 2–4, 128/256/512 tokens each). Use
  `mx.metal.get_active_memory()` to track peak. If peak stays under the
  M-series unified memory headroom, remove the call entirely.
- [ ] **B4-T2** If memory relief is genuinely needed, relocate
  `mx.clear_cache()` to run once per request — after generation completes
  and before audio encoding — instead of mid-loop.
- [ ] **B4-T3** Expose `FISH_MLX_CLEAR_CACHE_INTERVAL` env var (integer,
  read once at module import; `0` = disabled). **Default: `0` (disabled)**
  — behaviour change from current hard-coded `50`. Log the configured value
  at server startup. Document in `local_mlx/UPSTREAM_OPTIONS.md`.

---

## Bottleneck 5 — Audio encoding blocks the response path

**File:** `local_mlx/host_server.py:168–175`
**Impact:** Low-medium — `sf.write()` encodes the full PCM array into WAV/
FLAC/OGG synchronously after generation. For FLAC and OGG this adds
non-trivial CPU time (~50–200 ms depending on duration).

### Tasks

- [ ] **B5-T1** For WAV output, replace `sf.write()` with a direct
  `struct.pack` 44-byte header + int16 PCM bytes to eliminate the
  intermediate `BytesIO` copy and the C extension call overhead.
  **Critical:** the model output is `float32` in [-1, 1] (see
  `host_server.py:154`, `audio_np = np.asarray(audio, dtype=np.float32)`).
  `sf.write()` scales float→int16 automatically; a hand-rolled header path
  must scale explicitly or it emits near-silence. Compute `pcm` first, then
  size the header from `pcm.nbytes` (not the float array's `nbytes`):
  ```python
  import struct
  pcm = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)
  data_bytes = pcm.tobytes()
  header = struct.pack('<4sI4s4sIHHIIHH4sI',
      b'RIFF', 36 + len(data_bytes), b'WAVE',
      b'fmt ', 16, 1, 1, sample_rate,
      sample_rate * 2, 2, 16, b'data', len(data_bytes))
  return header + data_bytes, "audio/wav", timing
  ```
  Add a regression test asserting the new path is byte-identical (or
  perceptually equal) to the current `sf.write(..., format="WAV")` output for
  a fixed input — guards against the scaling/clipping bug above.
- [ ] **B5-T2** After B1-T3 async migration (if applicable): move FLAC/OGG
  encoding into a thread-pool executor (`concurrent.futures.
  ThreadPoolExecutor`, max_workers=2) so encoding does not block the event
  loop.
- [ ] **B5-T3** Add a startup log line confirming the default response
  format: `"Default response format: wav. FLAC/OGG incurs post-generation
  encoding latency (~50-200 ms)."` Update `README.md` to note this.

---

## Bottleneck 6 — No request batching

**File:** `local_mlx/host_server.py` (architecture-level)
**Impact:** Low (current single-user use) / High (multi-client use)

The batch inference path (`_generate_codes_for_text_batch_patched`,
`fish_speech_fastpath.py:260`) is defined in the fastpath patch but is not
wired into any HTTP endpoint. Batch mode shares the slow model's KV-cache
across multiple texts and parallelises the fast AR decoder, giving N×
throughput for multi-text requests.

> ⚠️ **GROUND TRUTH — the batch path is NOT runnable today (verified against
> the deployed `mlx_audio`):** `_generate_codes_for_text_batch_patched` calls
> two methods that are **defined nowhere** in `mlx_audio` (not in the model,
> not elsewhere in the package):
> - `self._prepare_batched_prompt_inputs(conversations)` (patch line 282)
> - `self._sample_semantic_batch(...)` (patch line 312)
>
> Upstream `fish_speech.py` has only the single-stream `_sample_semantic`
> (line 467) and `_generate_codes_for_batch` (line 504) — no batched prompt
> assembly and no batched sampler. Calling the patched batch fn as-is raises
> `AttributeError` on the first request. So B6 is **not** "wire up an existing
> path"; it is "**implement the missing batch primitives first**, then wire."
> Scope and effort are correspondingly higher than the original plan implied.

### Tasks

- [ ] **B6-T0** (Prerequisite — implement the missing primitives) Before any
  endpoint work, add the two helpers the patched batch fn depends on, as
  patched methods in `fish_speech_fastpath.py` (and register them in
  `apply_fish_speech_patch()` alongside the existing two):
  1. `_prepare_batched_prompt_inputs(self, conversations) -> (prompt, attention_mask)`
     — left-pad each conversation's `encode_for_inference` output to a common
     length, stack to `(batch, seq, num_cb+1)`, and build the matching
     `attention_mask` (`1` for real tokens, `0` for left-pad). Mirror the
     single-stream prompt assembly at `fish_speech.py:516–529`.
  2. `_sample_semantic_batch(self, logits, previous_semantic_tokens, top_p, top_k, temperature)`
     — vectorised form of `_sample_semantic` (`fish_speech.py:467`) operating
     over the batch dim, returning one semantic token per row. Honour the RAS
     window per-row.
  Gate on a correctness test: batch-of-1 output must match the single-stream
  path token-for-token (greedy/`temperature=0`) for the same input.
- [ ] **B6-T1** Add `POST /v1/audio/speech/batch` endpoint accepting:
  `{"input": ["text1", "text2", ...], ...}`. Route to the (now functional)
  batch path when batch size > 1. Return a JSON array of base64-encoded audio
  responses (one per text). Decode each batch row's codes with `_decode_codes`
  and encode per the B5/B2-T3 format rules.
- [ ] **B6-T2** Extend the existing `POST /v1/audio/speech` to auto-detect
  batch: if `input` is a JSON array instead of a string, route to the batch
  path. Single string input continues to use the single-stream path.
- [ ] **B6-T3** Implement a micro-batching window: after receiving a request,
  wait up to 20 ms for additional requests targeting the same model; group
  up to 4, dispatch the batch, return individual responses. Requires B1-T3
  async migration. Track and log batch utilisation rate.

---

## Bottleneck 7 — No graceful shutdown / model eviction

**File:** `local_mlx/host_server.py:314` (`httpd.serve_forever()`)
**Impact:** Medium — `SIGTERM`/`SIGINT` kills the process immediately
without draining in-flight requests. MLX model weights stay in unified
memory until the OS reaps the process. With B3/B4 optimizations reducing
per-step allocations, memory will grow monotonically under sustained load
without a reclaim path.

### Tasks

- [ ] **B7-T1** Add a signal handler for `SIGTERM` and `SIGINT`.
  **Threading constraint:** `httpd.serve_forever()` runs in the main thread,
  and `httpd.shutdown()` blocks until `serve_forever()` returns — so calling
  `shutdown()` directly from a main-thread signal handler deadlocks (it waits
  on the loop it is interrupting). Do **not** poke the name-mangled private
  `_BaseServer__shutdown_request`. Instead, either (a) run `serve_forever()`
  in a worker thread and `shutdown()` from the main thread's signal handler,
  or (b) keep `serve_forever()` in the main thread and have the handler only
  *set a flag*; call `httpd.shutdown()` from a short-lived dedicated thread the
  handler spawns. Whichever shape, the drain/evict sequence is:
  1. Stop accepting new connections.
  2. Drain in-flight requests (wait up to 30 s for active generation loops,
     using the B1-T4 in-flight counter to know when depth hits 0).
  3. Evict the model: `self.model = None` + `mx.clear_cache()`.
  4. `httpd.shutdown()` (from a non-serving thread) and exit with code 0.
- [ ] **B7-T2** Log the shutdown reason (signal name, drain timeout reached,
  clean exit) at INFO level before exiting.

---

## Bottleneck 8 — Health endpoint reports incomplete state

**File:** `local_mlx/host_server.py:217–229` (`/health`)
**Impact:** Low — `/health` returns `model_loaded: true/false` but never
reports operational metrics. A stuck server looks identical to a healthy
one from the health endpoint.

### Tasks

- [ ] **B8-T1** Extend the `/health` response body to include:
  ```json
  {
    "ok": true,
    "model_loaded": true,
    "uptime_seconds": 1234.5,
    "last_request": {
      "rtf": 0.35,
      "gen_seconds": 1.05,
      "audio_seconds": 3.0,
      "semantic_tokens": 64
    },
    "last_error": null,
    "in_flight_requests": 2
  }
  ```
  Fields: `uptime_seconds` (monotonic since `FishMLXServer.__init__`),
  `last_request` (copy of `self.last_timing` from most recent `synthesize()`),
  `last_error` (most recent exception `{message, type, timestamp}` or
  `null`), `in_flight_requests` (counter from B1-T4).

---

## Ordering & Dependencies

| Order | Bottleneck | Depends on | Reason |
|-------|-----------|------------|--------|
| 1 | B5 (WAV encoding) | — | Low effort, independent, real correctness fix (float→int16) |
| 2 | B8 (health metrics) | — | Very low effort, aids debugging all other bottlenecks |
| 3 | B3 (pre-allocation) | — | Low effort, independent micro-optimisation |
| 4 | B2 (segment streaming) | — | Latency reduction *for multi-segment input only*; segment-level, sync server |
| 5 | B1-T1 (concurrency stress test) | — | Investigation starts early; result gates B1-T2 vs B1-T3 |
| 6 | B7 (graceful shutdown) | — | Independent, operational hygiene |
| 7 | B1-T2/T3 (lock removal / async) | B2, B1-T1 | Must follow B2 on current server; gates on T1 results |
| 8 | B6 (batching) | B3, B6-T0 | Must first implement missing batch primitives (B6-T0), then wire into HTTP |
| 9 | B4 (clear_cache) | B6 | Dormant until B6 activates the batch path; do it *within* the B6 pass |

> **Dependency note:** B4 and B6 are not mutually dependent. B4 only matters
> once B6 wires `_generate_codes_for_text_batch_patched` into an HTTP endpoint
> (today nothing calls it, so the `clear_cache()` at line 406 never fires). B4
> is therefore a sub-task *inside* the B6 implementation pass, not a separate
> milestone that B6 waits on. Earlier revisions listed `B4 → B6` and `B6 → B4`
> simultaneously, which is a cycle; the correct edge is `B4 → B6` only.

## Priority order

| # | Task(s) | Expected gain | Effort |
|---|---------|---------------|--------|
| 1 | B5-T1..T3 (WAV encoding) | Correctness (float→int16) + −1–5% latency | Very Low |
| 2 | B8-T1 (health metrics) | Operational visibility | Very Low |
| 3 | B3-T1..T3 (pre-allocation) | −5–15% token latency | Low |
| 4 | B2-T0..T5 (segment streaming) | TTFB win on multi-segment input only; none for short utterances | Medium |
| 5 | B1-T1 (concurrency stress test) | Gates lock removal | Low |
| 6 | B7-T1..T2 (graceful shutdown) | Operational reliability | Low |
| 7 | B1-T2..T4 (lock removal / async) | +parallelism | Medium |
| 8 | B6-T0..T3 (batch primitives + batching) + B4-T1..T3 (clear_cache, folded in) | ×N throughput | High |

**Execution plan:** Start with the three low-risk, independent wins — **B5**
(also a real correctness fix: the model emits `float32`, not int16), **B8**,
and **B3** — they are single-file changes with no shared state and need no
upstream investigation. Then take on **B2** (segment-level streaming):
re-read the B2 GROUND TRUTH box first — `stream=True` is unsupported and the
generator yields per-segment, so the win is real only for multi-segment
input. Begin **B1-T1** (concurrency stress test) in parallel with B2; its
result determines whether B1-T2 (simple lock removal) or B1-T3 (full async
migration) is correct. **B6** is the largest item and is gated on **B6-T0**
(implementing the two missing batch primitives — `_prepare_batched_prompt_inputs`
and `_sample_semantic_batch` — which do not exist upstream); do not start B6
expecting a ready-made batch path. **B4** is dormant until **B6** activates
the batch path — fold it into the B6 pass, not before.

## Verification — re-run after each bottleneck

```bash
pytest -x --tb=short tests/

python local_mlx/profile_generation.py \
  --model-path checkpoints/fish-audio-s2-pro-8bit-mlx-normalized

curl -s -X POST http://127.0.0.1:8881/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello world.","model":"tts-1","voice":"default"}' \
  -o /tmp/smoke.wav && file /tmp/smoke.wav
```

## Out of scope

- Production deployment (Docker Compose healthchecks, k8s probes)
- Multi-tenant authentication / API key management
- Rate limiting beyond `X-Queue-Depth` response header
- Model quantisation level tuning (8-bit is the assumed baseline)
- **True per-frame / per-token audio streaming.** The deployed model's
  `generate()` decodes a whole segment's codes at once (`_decode_codes`,
  `fish_speech.py:602`) and has no incremental-decode path
  (`stream=True` → `NotImplementedError`, line 630). Sub-segment streaming
  would require implementing sliding-window codec decode inside the fastpath
  patch and is a separate project, not part of B2. B2 delivers *segment-level*
  streaming only.

---

## Environment of record (where these facts were verified)

All upstream line numbers and behaviours above were verified against the
interpreter the deployment actually uses (see `start_mlx_local.sh:9`,
`local_mlx/com.op.fish-mlx-host.plist:10`,
`local_mlx/patches/apply_patch.sh:5`):

- **Python / venv:** `/Users/op/fish-speech-int4-patch/.venv-mlx/bin/python` (3.11)
- **Upstream model:** `…/.venv-mlx/lib/python3.11/site-packages/mlx_audio/tts/models/fish_qwen3_omni/fish_speech.py`

Other `mlx_audio` copies exist on this machine (under `locateanything`,
`.unsloth`, `tts-venv`, `.cache/uv`). **Ignore them** — they are not what the
host server runs. If you upgrade `mlx_audio`, re-verify the two GROUND TRUTH
boxes (B2 streaming, B6 batch primitives): both depend on upstream internals
that may change between versions.