#!/usr/bin/env python3
"""FastAPI OpenAI-compatible /v1 TTS API for local MLX Fish Audio S2 Pro.

This is a thin FastAPI wrapper around local_mlx.host_server.FishMLXServer.
It keeps the house pattern used by the other local TTS services:
  - process starts light
  - model is loaded on first /v1/audio/speech request
  - model is evicted after an idle window
  - OpenAI-compatible /v1/audio/speech and /v1/models endpoints
"""
from __future__ import annotations

import argparse
import gc
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from local_mlx.host_server import DEFAULT_MAX_TOKENS, MODEL_ID, OPENAI_MODELS, FishMLXServer


class SpeechRequest(BaseModel):
    model: str = Field(default="tts-1")
    input: str
    voice: str = Field(default="default")
    response_format: Literal["wav", "flac", "ogg"] = Field(default="wav")
    speed: float = Field(default=1.0)
    language: str | None = None
    lang_code: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    greedy: bool | None = None
    cfg_scale: float | None = None
    ddpm_steps: int | None = None
    verbose: bool | None = None


app = FastAPI(title="Fish S2 Pro MLX TTS", version="1.0.0")

_fish: FishMLXServer | None = None
_last_access = 0.0
_idle_timeout_s = float(os.environ.get("FISH_MLX_IDLE_TIMEOUT_S", "300"))
_evict_interval_s = float(os.environ.get("FISH_MLX_EVICT_INTERVAL_S", "15"))
_state_lock = threading.Lock()
_evict_thread_started = False


def _touch() -> None:
    global _last_access
    _last_access = time.monotonic()


def _idle_seconds() -> float | None:
    if _last_access <= 0:
        return None
    return round(time.monotonic() - _last_access, 3)


def _server() -> FishMLXServer:
    if _fish is None:
        raise RuntimeError("Fish server has not been initialised")
    return _fish


def _evict_loop() -> None:
    while True:
        time.sleep(_evict_interval_s)
        fish = _fish
        if fish is None or fish.model is None:
            continue
        if _last_access <= 0 or time.monotonic() - _last_access < _idle_timeout_s:
            continue
        if fish._snapshot_in_flight() != 0:  # noqa: SLF001 - wrapper over local class
            continue
        with _state_lock:
            if fish.model is not None and time.monotonic() - _last_access >= _idle_timeout_s:
                print(
                    f"[fish-fastapi] idle for {time.monotonic() - _last_access:.1f}s; evicting model",
                    flush=True,
                )
                fish.evict_model()
                try:
                    mx.clear_cache()
                except Exception as exc:  # best effort
                    print(f"[fish-fastapi] mx.clear_cache failed: {exc!r}", flush=True)
                gc.collect()


@app.on_event("startup")
def startup() -> None:
    global _fish, _evict_thread_started
    model_path = Path(os.environ["FISH_MLX_MODEL_PATH"]).expanduser().resolve()
    lazy = os.environ.get("FISH_MLX_LOAD_MODE", "lazy").lower() != "eager"
    warmup = os.environ.get("FISH_MLX_WARMUP", "0") in {"1", "true", "yes"}
    _fish = FishMLXServer(model_path=model_path, lazy=lazy, warmup=warmup)
    print(
        f"[fish-fastapi] ready model_path={model_path} lazy={lazy} warmup={warmup} "
        f"idle_timeout_s={_idle_timeout_s}",
        flush=True,
    )
    if not _evict_thread_started:
        threading.Thread(target=_evict_loop, daemon=True, name="fish-fastapi-evict").start()
        _evict_thread_started = True


@app.get("/health")
def health() -> dict[str, Any]:
    fish = _server()
    data = fish.snapshot_health()
    data.update(
        {
            "backend": "fish-s2-pro-mlx-fastapi",
            "api": "openai-compatible /v1/audio/speech",
            "lazy_loading": True,
            "idle_timeout_s": _idle_timeout_s,
            "idle_seconds": _idle_seconds(),
            "port": int(os.environ.get("PORT", "8882")),
        }
    )
    return data


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1710000000,
                "owned_by": "local",
                "use_case": "hq_offline" if model_id in {MODEL_ID, "s2-pro-mlx"} else "compat",
            }
            for model_id in OPENAI_MODELS
        ],
    }


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    fish = _server()
    _touch()
    fish._inc_in_flight()  # noqa: SLF001 - wrapper over local class
    try:
        payload = req.model_dump(exclude_none=True)
        audio, media_type, timing = fish.synthesize(payload)
        headers = {
            "X-Queue-Depth": str(fish._snapshot_in_flight()),  # noqa: SLF001
        }
        if timing:
            headers.update(
                {
                    "X-Gen-Seconds": f"{timing['gen_seconds']:.6f}",
                    "X-Audio-Seconds": f"{timing['audio_seconds']:.6f}",
                    "X-RTF": f"{timing['rtf']:.6f}",
                    "X-Semantic-Tokens": str(timing["semantic_tokens"]),
                    "X-Max-Tokens-Effective": str(timing["max_tokens_effective"]),
                }
            )
        return Response(content=audio, media_type=media_type, headers=headers)
    except Exception as exc:
        fish._record_error(exc)  # noqa: SLF001
        raise HTTPException(status_code=400, detail={"message": str(exc), "type": exc.__class__.__name__}) from exc
    finally:
        _touch()
        fish._dec_in_flight()  # noqa: SLF001


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8882")))
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "FISH_MLX_MODEL_PATH",
            # Default to the ASR-verified 8-bit normalized checkpoint (the proven
            # working build; the 4-bit conversions decode to noise). Relative path
            # matches host_server.py so both servers default to the same model.
            "checkpoints/fish-audio-s2-pro-8bit-mlx-normalized",
        ),
    )
    args = parser.parse_args()
    os.environ["FISH_MLX_MODEL_PATH"] = args.model_path
    os.environ["PORT"] = str(args.port)
    import uvicorn

    uvicorn.run("local_mlx.fastapi_v1_server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
