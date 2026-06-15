# Local MLX Fish S2 Pro Deployment

HQ/offline Fish S2 MLX — **not** for live conversation. Use VibeVoice or Chatterbox
(`tts-multimodel-api`, default `vibe-realtime-8bit`) for talk mode and low latency.

This deployment uses `cs2764/fish-audio-s2-pro-8bit-mlx`, normalized into
`checkpoints/fish-audio-s2-pro-8bit-mlx-normalized`.

## Model selection — use 8-bit, never 4-bit

The **8-bit normalized** checkpoint is the only ASR-verified working build and is
the default for both servers. **Do not use the 4-bit conversions**
(`ekryski/fish-audio-s2-pro-4bit`, `majentik/fishaudio-s2-pro-MLX-4bit`): both
decode to **near-silent noise**, confirmed with the ASR on `:5093` (empty
transcription for every 4-bit file; the 8-bit control transcribes correctly). The
4-bit quantization degrades the text/semantic model itself, so even keeping the
codec at full precision (as `majentik` does, via a separate `codec.pth`) does not
recover intelligible speech.

### Always ASR-verify output

"It loaded and emitted samples" is **not** proof of intelligible speech. After any
synthesis, corroborate with the macOS speech-server ASR on `:5093`:

```bash
curl -s -X POST http://127.0.0.1:5093/v1/audio/transcriptions \
  -F file=@out.wav -F model=whisper-1 -F response_format=text
```

An empty transcription (or peak amplitude ≲ 0.02) means the audio is noise/silence
— the 4-bit failure mode.

Docker Desktop on Apple Silicon runs Linux containers and cannot access Apple
Metal/MLX directly:

- macOS host: MLX model on `127.0.0.1:8881` (eager load + warmup + fastpath patch)
- Docker proxy: OpenAI-compatible API on `127.0.0.1:8880` → `host.docker.internal:8881`

## Paths (this machine)

| Item | Path |
|---|---|
| Python venv | `/Users/op/fish-speech-int4-patch/.venv-mlx/bin/python` |
| Model | `/Users/op/fish-speech-int4-patch/checkpoints/fish-audio-s2-pro-8bit-mlx-normalized` |
| Deploy repo | `/Users/op/fish-s2-pro-mlx-local-deploy` |

The LaunchAgent plist and benchmark commands below use these paths.

## Setup

```bash
cd /Users/op/fish-s2-pro-mlx-local-deploy
/Users/op/fish-speech-int4-patch/.venv-mlx/bin/python local_mlx/normalize_cs2764_checkpoint.py \
  /Users/op/fish-speech-int4-patch/checkpoints/fish-audio-s2-pro-8bit-mlx \
  /Users/op/fish-speech-int4-patch/checkpoints/fish-audio-s2-pro-8bit-mlx-normalized
./start_mlx_local.sh
```

The generation fastpath patch is applied automatically at host startup via
`local_mlx/patches/fish_speech_fastpath.py` (runtime monkey-patch, no site-packages edit).
To verify manually:

```bash
PYTHONPATH=/Users/op/fish-s2-pro-mlx-local-deploy \
  /Users/op/fish-speech-int4-patch/.venv-mlx/bin/python local_mlx/patches/apply_patch.sh
```

## Routing (which backend when)

| Use case | Backend | Endpoint |
|---|---|---|
| Talk / live / low latency | VibeVoice (`vibe-realtime-8bit`) | `tts-multimodel-api` :8000 |
| Chatterbox multilingual | `chatterbox-*` | `tts-multimodel-api` :8000 |
| Clone / narration / HQ offline | Fish S2 MLX | host `:8881` or Docker proxy `:8880` |
| OpenAI-compatible speech API | Fish S2 via proxy | `POST /v1/audio/speech` on `:8880` |

In `tts-multimodel-api`:

- Default model: `vibe-realtime-8bit` (live conversation)
- HQ alias: `fish-s2-pro-quality` (same weights as `fish-s2-pro-4bit`, capped at 256 tokens)
- `GET /models` and `GET /health` document `use_case` and `fish_s2_max_tokens`

Fish S2 MLX does **not** support streaming (`stream=True` raises in mlx-audio).

## Detached LaunchAgent

```bash
cp local_mlx/com.op.fish-mlx-host.plist ~/Library/LaunchAgents/com.op.fish-mlx-host.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.op.fish-mlx-host.plist
launchctl enable gui/$(id -u)/com.op.fish-mlx-host
launchctl kickstart -k gui/$(id -u)/com.op.fish-mlx-host
docker compose -f compose.mlx.yml up -d --build
```

Host defaults: `--no-lazy`, `--warmup`, `FISH_MLX_MAX_TOKENS=256`, `PYTHONPATH` set to deploy repo.

Optional request flags on `:8881`:

- `greedy: true` — sets `temperature=0` and uses compiled argmax on the fast residual path
- Response headers: `X-Gen-Seconds`, `X-Audio-Seconds`, `X-RTF`, `X-Semantic-Tokens`, `X-Max-Tokens-Effective`, `X-Queue-Depth` (current in-flight synthesis count)

## Response format and encoding latency

Default `response_format` is `wav`. The host encodes WAV with a hand-rolled
`struct.pack` 44-byte RIFF/WAVE header + int16 PCM (B5-T1) — no
`BytesIO` round-trip, no `soundfile` C-extension call. Audio is
perceptually identical to `sf.write(..., format="WAV")` (max ~2 LSB
difference on full-scale random audio, see
`tests/test_b5t1_wav_encoder.py`).

`response_format=flac` and `response_format=ogg` still go through
`soundfile.write` and incur **~50–200 ms of post-generation encoding
latency** on top of the model's generation time. If you need TTFB, stick
to `wav`. (See `TODO_BOTTLENECKS.md` B5-T2 / B2-T3 — making FLAC/OGG
streamable is a larger change, tracked in the bottleneck plan, not
shipped here.)

## Health endpoint

`GET /health` returns the extended payload (B8-T1):

```json
{
  "ok": true,
  "backend": "mlx-host",
  "model_loaded": true,
  "hq_offline_only": true,
  "default_max_tokens": 256,
  "patch_applied": true,
  "uptime_seconds": 1234.5,
  "in_flight_requests": 0,
  "last_request": {
    "gen_seconds": 1.05,
    "audio_seconds": 3.0,
    "rtf": 0.35,
    "semantic_tokens": 64,
    "max_tokens_requested": 256,
    "max_tokens_effective": 256
  },
  "last_error": null
}
```

A stuck or repeatedly-failing instance is no longer silent from the
health endpoint.

## Smoke checks

```bash
curl http://127.0.0.1:8881/health
curl -i -X POST http://127.0.0.1:8881/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"fish-audio-s2-pro-8bit-mlx","input":"Hello.","max_tokens":8}' \
  --output /tmp/fish-s2-pro-mlx.wav
```

Via Docker proxy:

```bash
curl http://127.0.0.1:8880/health
```

## RTF benchmark

```bash
PYTHONPATH=/Users/op/fish-s2-pro-mlx-local-deploy \
  /Users/op/fish-speech-int4-patch/.venv-mlx/bin/python local_mlx/profile_generation.py \
  --model-path /Users/op/fish-speech-int4-patch/checkpoints/fish-audio-s2-pro-8bit-mlx-normalized
```

Greedy mode (compiled fast-path argmax):

```bash
PYTHONPATH=/Users/op/fish-s2-pro-mlx-local-deploy \
  /Users/op/fish-speech-int4-patch/.venv-mlx/bin/python local_mlx/profile_generation.py \
  --model-path /Users/op/fish-speech-int4-patch/checkpoints/fish-audio-s2-pro-8bit-mlx-normalized \
  --greedy \
  --output local_mlx/benchmarks/rtf_report_greedy.json
```

Reports: `local_mlx/benchmarks/rtf_report.json` (stochastic) and `rtf_report_greedy.json`.

### Acceptance gates

| Case | RTF threshold | Baseline (pre-patch) |
|---|---:|---:|
| `Hello.` (~8 tokens) | &lt; 3.0 | 3.99 |
| Short sentence (~64 tokens) | &lt; 1.6 | 2.07 |
| Longer sample (~150 tokens) | &lt; 1.5 | 2.08 |

The script exits non-zero if any gate fails. After the fastpath patch, the hello case
typically passes; longer clips may still exceed gates (~2.0 RTF) — see below.

## If RTF is still too high

See [UPSTREAM_OPTIONS.md](UPSTREAM_OPTIONS.md) for upstream PR, Swift prototype, or CUDA server paths.

## OpenAI `/v1` FastAPI server (`:8882`)

`fastapi_v1_server.py` is a thin FastAPI wrapper around the same `FishMLXServer`
core, for setups that want a light footprint with lazy load + idle eviction. It
defaults to the 8-bit normalized checkpoint (relative path, matching
`host_server.py`).

```bash
python local_mlx/fastapi_v1_server.py --host 127.0.0.1 --port 8882
# model loads on the first /v1/audio/speech request, evicts after idle
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/audio/speech` (same OpenAI
schema and `X-RTF`/`X-Gen-Seconds`/… timing headers as `:8881`).

Environment variables:

| Var | Default | Meaning |
|---|---|---|
| `FISH_MLX_MODEL_PATH` | `checkpoints/fish-audio-s2-pro-8bit-mlx-normalized` | Model snapshot dir. |
| `FISH_MLX_LOAD_MODE` | `lazy` | `lazy` or `eager`. |
| `FISH_MLX_WARMUP` | `0` | `1` to warm caches at startup. |
| `FISH_MLX_IDLE_TIMEOUT_S` | `300` | Evict model after this idle window. |
| `FISH_MLX_EVICT_INTERVAL_S` | `15` | Eviction-check interval. |
| `PORT` | `8882` | Listen port. |

Difference vs `host_server.py` (`:8881`): that server defaults to **eager load +
warmup** (low first-request latency, the reference perf path); this one defaults to
**lazy + evict** (loads on demand). For a warmed comparison the `:8881` server gives
synthesis **RTF ≈ 3.8** (non-greedy HTTP path, ~5.7 tok/s); the `profile_generation.py`
greedy fast-path gates above are lower because they exclude HTTP + stochastic sampling.

## Checkpoint normalization

The cs2764 safetensors use already-sanitized keys; `mlx-audio@fish-audio-s2` expects
upstream keys. The normalizer rewrites to `text_model.model.*` and `audio_decoder.*`
so `strict=True` load works.
