# Fish S2 MLX — upstream and alternative backends

If the local fastpath patch still leaves RTF above ~1.5 on ~100-token clips, use one of these paths instead of more deployment tuning.

## 1. Upstream PR to mlx-audio

Target branch: `lucasnewman/mlx-audio@fish-audio-s2`

Changes to contribute:

- Tensor RAS in `_sample_semantic` (avoid Python list + `.item()` in the hot path)
- Single `mx.eval()` per semantic token after the fast residual codebook loop
- Fix `_generate_codes_for_batch()` to refresh `hidden_state` each slow step
- Optional `@mx.compile` greedy residual sampling

The same loop exists in `mlx-audio-swift` (`FishSpeechModel.swift`); a Python fix should be mirrored there.

## 2. Swift native prototype

Repo: `/Users/op/s2pro-native-prototype`

The Swift port uses the same DualAR structure. Worth profiling only if eval/sync is reduced in Swift in parallel with the Python patch.

## 3. CUDA Fish Speech server (official throughput)

This repo’s Docker `server` profile + `COMPILE=1` is what backs the published RTF ~0.19 claims. Use for Linux/GPU batch HQ; keep MLX on Apple Silicon for local convenience.

## 4. Do not pursue

- Docker-side MLX on Apple Silicon
- HTTP/WAV/codec micro-optimizations
- Fish S2 MLX for live conversation (use VibeVoice/Chatterbox via `tts-multimodel-api`)
