# Local MLX Fish S2 Pro Deployment

This deployment uses the requested `cs2764/fish-audio-s2-pro-8bit-mlx` model
snapshot, normalized into `checkpoints/fish-audio-s2-pro-8bit-mlx-normalized`.

Docker Desktop on Apple Silicon runs Linux containers and cannot access Apple
Metal/MLX directly. The working local shape is therefore:

- macOS host process: loads the MLX model on `127.0.0.1:8881`
- Docker container: exposes the local OpenAI-compatible API on `127.0.0.1:8880`
  and proxies to `host.docker.internal:8881`

Start both:

```bash
cd /Users/op/fish-speech-int4-patch
.venv-mlx/bin/python local_mlx/normalize_cs2764_checkpoint.py \
  checkpoints/fish-audio-s2-pro-8bit-mlx \
  checkpoints/fish-audio-s2-pro-8bit-mlx-normalized
./start_mlx_local.sh
```

Detached start:

```bash
cd /Users/op/fish-speech-int4-patch
cp local_mlx/com.op.fish-mlx-host.plist ~/Library/LaunchAgents/com.op.fish-mlx-host.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.op.fish-mlx-host.plist
launchctl enable gui/$(id -u)/com.op.fish-mlx-host
launchctl kickstart -k gui/$(id -u)/com.op.fish-mlx-host
docker compose -f compose.mlx.yml up -d --build
```

Stop detached services:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.op.fish-mlx-host.plist
docker compose -f compose.mlx.yml down
```

Smoke checks:

```bash
curl http://127.0.0.1:8880/health
curl http://127.0.0.1:8880/v1/models
curl -X POST http://127.0.0.1:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"fish-audio-s2-pro-8bit-mlx","input":"Hello.","response_format":"wav","lang_code":"en","max_tokens":8}' \
  --output /tmp/fish-s2-pro-mlx.wav
```

Root cause fixed here: the cs2764 safetensors use already-sanitized weight keys,
but `lucasnewman/mlx-audio@fish-audio-s2` expects upstream keys and sanitizes
them at load time. Loading the raw cs2764 snapshot with `strict=False` silently
dropped model weights and made synthesis hang. The normalizer rewrites the keys
to `text_model.model.*` and `audio_decoder.*`, after which the server loads with
`strict=True`.

Verified on this Mac: health/model-list work through Docker, and an 8-token
`/v1/audio/speech` request returned a valid 44.1 kHz WAV in 4 seconds.
