#!/usr/bin/env python3
"""OpenAI-compatible local API for MLX Fish Audio S2 Pro.

Docker Desktop Linux containers cannot access Apple Metal/MLX directly, so this
server runs on macOS and is fronted by the Docker proxy in this directory.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model


MODEL_ID = "fish-audio-s2-pro-8bit-mlx"
OPENAI_MODELS = ("tts-1", "tts-1-hd", MODEL_ID, "s2-pro-mlx")
OPENAI_VOICES = {
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "default",
    "",
}


class FishMLXServer:
    def __init__(self, model_path: Path, lazy: bool):
        self.model_path = model_path
        self.lazy = lazy
        self.model = None
        self.lock = threading.Lock()
        self.loaded_at: float | None = None

    def ensure_model(self):
        if self.model is None:
            with self.lock:
                if self.model is None:
                    self.model = load_model(self.model_path, lazy=self.lazy, strict=True)
                    self.loaded_at = time.time()
        return self.model

    def synthesize(self, payload: dict[str, Any]) -> tuple[bytes, str]:
        text = str(payload.get("input") or payload.get("text") or "").strip()
        if not text:
            raise ValueError("Missing required `input` text.")

        requested_model = str(payload.get("model") or MODEL_ID)
        if requested_model not in OPENAI_MODELS:
            raise ValueError(f"Unsupported model `{requested_model}`.")

        voice = str(payload.get("voice") or "default")
        if voice not in OPENAI_VOICES:
            raise ValueError(f"Unsupported voice `{voice}`.")

        response_format = str(payload.get("response_format") or "wav").lower()
        if response_format not in {"wav", "flac", "ogg"}:
            raise ValueError("Supported response_format values: wav, flac, ogg.")

        model = self.ensure_model()
        gen_kwargs = {
            "text": text,
            "voice": "default",
            "speed": float(payload.get("speed") or 1.0),
            "lang_code": str(payload.get("lang_code") or payload.get("language") or "en"),
            "temperature": float(payload.get("temperature") or 0.7),
            "max_tokens": int(payload.get("max_tokens") or 1200),
            "verbose": bool(payload.get("verbose", False)),
            "stream": False,
        }
        if payload.get("cfg_scale") is not None:
            gen_kwargs["cfg_scale"] = float(payload["cfg_scale"])
        if payload.get("ddpm_steps") is not None:
            gen_kwargs["ddpm_steps"] = int(payload["ddpm_steps"])

        chunks = []
        sample_rate = getattr(model, "sample_rate", 24000)
        with self.lock:
            for result in model.generate(**gen_kwargs):
                chunks.append(result.audio)
                sample_rate = result.sample_rate

        if not chunks:
            raise RuntimeError("Model returned no audio.")

        audio = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=0)
        audio_np = np.asarray(audio, dtype=np.float32)
        out = io.BytesIO()
        sf.write(out, audio_np, sample_rate, format=response_format.upper())
        media_type = {
            "wav": "audio/wav",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
        }[response_format]
        return out.getvalue(), media_type


class Handler(BaseHTTPRequestHandler):
    server_version = "FishMLXHTTP/1.0"

    def _json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _bytes(self, status: int, body: bytes, media_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/health":
            model_loaded = self.server.fish.model is not None  # type: ignore[attr-defined]
            self._json(HTTPStatus.OK, {"ok": True, "backend": "mlx-host", "model_loaded": model_loaded})
            return
        if self.path == "/v1/models":
            self._json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {"id": model_id, "object": "model", "created": 1710000000, "owned_by": "local"}
                        for model_id in OPENAI_MODELS
                    ],
                },
            )
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found"}})

    def do_POST(self) -> None:
        if self.path != "/v1/audio/speech":
            self._json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found"}})
            return
        try:
            audio, media_type = self.server.fish.synthesize(self._read_json())  # type: ignore[attr-defined]
            self._bytes(HTTPStatus.OK, audio, media_type)
        except Exception as exc:
            self._json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": str(exc), "type": exc.__class__.__name__}},
            )

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8881)
    parser.add_argument(
        "--model-path",
        default="checkpoints/fish-audio-s2-pro-8bit-mlx",
        help="Local path to the downloaded MLX model snapshot.",
    )
    parser.add_argument("--lazy", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.fish = FishMLXServer(model_path=model_path, lazy=args.lazy)  # type: ignore[attr-defined]
    print(f"Fish MLX host server listening on http://{args.host}:{args.port}", flush=True)
    print(f"Model path: {model_path}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
