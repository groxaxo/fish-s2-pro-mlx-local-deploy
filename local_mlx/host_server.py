#!/usr/bin/env python3
"""OpenAI-compatible local API for MLX Fish Audio S2 Pro.

Docker Desktop Linux containers cannot access Apple Metal/MLX directly, so this
server runs on macOS and is fronted by the Docker proxy in this directory.
"""

from __future__ import annotations

import argparse
import sys
import io
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mlx.core as mx
import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

from local_mlx.patches.fish_speech_fastpath import apply_fish_speech_patch


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
DEFAULT_MAX_TOKENS = int(os.environ.get("FISH_MLX_MAX_TOKENS", "256"))
DEFAULT_WARMUP_TEXT = os.environ.get("FISH_MLX_WARMUP_TEXT", "Hi.")


class FishMLXServer:
    def __init__(self, model_path: Path, lazy: bool, warmup: bool):
        self.model_path = model_path
        self.lazy = lazy
        self.warmup = warmup
        self.model = None
        self.lock = threading.Lock()
        self.loaded_at: float | None = None
        self.started_at = time.time()
        self.last_timing: dict[str, float | int] = {}
        self.last_error: dict[str, Any] | None = None
        # In-flight synthesis counter (B1-T4). Mutated only under
        # ``self.queue_lock`` — ``+=`` on a bare int is not GIL-safe across
        # threads, so the dedicated lock is mandatory, not stylistic.
        self.in_flight = 0
        self.queue_lock = threading.Lock()

    def _inc_in_flight(self) -> None:
        with self.queue_lock:
            self.in_flight += 1

    def _dec_in_flight(self) -> None:
        with self.queue_lock:
            self.in_flight -= 1

    def _snapshot_in_flight(self) -> int:
        with self.queue_lock:
            return self.in_flight

    def _record_error(self, exc: BaseException) -> None:
        """Record the most recent error for /health visibility (B8-T1)."""
        self.last_error = {
            "type": exc.__class__.__name__,
            "message": str(exc),
            "timestamp": time.time(),
        }

    def snapshot_health(self) -> dict[str, Any]:
        """Build the extended /health payload (B8-T1)."""
        return {
            "ok": True,
            "backend": "mlx-host",
            "model_loaded": self.model is not None,
            "hq_offline_only": True,
            "default_max_tokens": DEFAULT_MAX_TOKENS,
            "patch_applied": True,
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "in_flight_requests": self._snapshot_in_flight(),
            "last_request": dict(self.last_timing) if self.last_timing else None,
            "last_error": self.last_error,
        }

    def ensure_model(self):
        if self.model is None:
            with self.lock:
                if self.model is None:
                    apply_fish_speech_patch()
                    self.model = load_model(self.model_path, lazy=self.lazy, strict=True)
                    self.loaded_at = time.time()
                    if self.warmup:
                        self._run_warmup()
        return self.model

    def _run_warmup(self) -> None:
        """Run warmup synthesis. Caller must hold ``self.lock``."""
        print("Running Fish MLX warmup synthesis...", flush=True)
        started = time.perf_counter()
        for _ in self.model.generate(
            text=DEFAULT_WARMUP_TEXT,
            max_tokens=8,
            temperature=0.0,
            verbose=False,
            stream=False,
        ):
            pass
        mx.eval(mx.array(0))
        elapsed = time.perf_counter() - started
        print(f"Warmup complete in {elapsed:.3f}s", flush=True)

    def _effective_max_tokens(self, model: Any, text: str, requested: int) -> int:
        requested = max(1, min(requested, DEFAULT_MAX_TOKENS))
        if model.tokenizer is None:
            return requested
        text_tokens = len(model.tokenizer.encode(text))
        budget = min(requested, max(32, text_tokens * 12))
        return budget

    def synthesize(self, payload: dict[str, Any]) -> tuple[bytes, str, dict[str, float | int]]:
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

        greedy = bool(payload.get("greedy", False))
        temperature = float(payload.get("temperature") or 0.7)
        if greedy:
            temperature = 0.0

        model = self.ensure_model()
        requested_max_tokens = int(payload.get("max_tokens") or DEFAULT_MAX_TOKENS)
        effective_max_tokens = self._effective_max_tokens(model, text, requested_max_tokens)

        gen_kwargs = {
            "text": text,
            "voice": "default",
            "speed": float(payload.get("speed") or 1.0),
            "lang_code": str(payload.get("lang_code") or payload.get("language") or "en"),
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
            "verbose": bool(payload.get("verbose", False)),
            "stream": False,
        }
        if payload.get("cfg_scale") is not None:
            gen_kwargs["cfg_scale"] = float(payload["cfg_scale"])
        if payload.get("ddpm_steps") is not None:
            gen_kwargs["ddpm_steps"] = int(payload["ddpm_steps"])

        chunks = []
        sample_rate = getattr(model, "sample_rate", 24000)
        semantic_tokens = 0
        started = time.perf_counter()
        with self.lock:
            for result in model.generate(**gen_kwargs):
                chunks.append(result.audio)
                sample_rate = result.sample_rate
                semantic_tokens += int(getattr(result, "token_count", 0) or 0)

        if not chunks:
            raise RuntimeError("Model returned no audio.")

        audio = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=0)
        audio_np = np.asarray(audio, dtype=np.float32)
        gen_seconds = max(time.perf_counter() - started, 1e-6)
        audio_seconds = float(audio_np.shape[0]) / float(sample_rate)
        rtf = gen_seconds / max(audio_seconds, 1e-6)

        self.last_timing = {
            "gen_seconds": gen_seconds,
            "audio_seconds": audio_seconds,
            "rtf": rtf,
            "semantic_tokens": semantic_tokens,
            "max_tokens_requested": requested_max_tokens,
            "max_tokens_effective": effective_max_tokens,
        }

        out = io.BytesIO()
        sf.write(out, audio_np, sample_rate, format=response_format.upper())
        media_type = {
            "wav": "audio/wav",
            "flac": "audio/flac",
            "ogg": "audio/ogg",
        }[response_format]
        return out.getvalue(), media_type, self.last_timing


class Handler(BaseHTTPRequestHandler):
    server_version = "FishMLXHTTP/1.0"

    def _json(self, status: int, body: dict[str, Any], queue_depth: int | None = None) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        if queue_depth is not None:
            self.send_header("X-Queue-Depth", str(queue_depth))
        self.end_headers()
        self.wfile.write(encoded)

    def _bytes(
        self,
        status: int,
        body: bytes,
        media_type: str,
        timing: dict[str, float | int] | None = None,
        queue_depth: int | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(body)))
        if queue_depth is not None:
            self.send_header("X-Queue-Depth", str(queue_depth))
        if timing:
            self.send_header("X-Gen-Seconds", f"{timing['gen_seconds']:.6f}")
            self.send_header("X-Audio-Seconds", f"{timing['audio_seconds']:.6f}")
            self.send_header("X-RTF", f"{timing['rtf']:.6f}")
            self.send_header("X-Semantic-Tokens", str(timing["semantic_tokens"]))
            self.send_header(
                "X-Max-Tokens-Effective", str(timing["max_tokens_effective"])
            )
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/health":
            fish = self.server.fish  # type: ignore[attr-defined]
            self._json(HTTPStatus.OK, fish.snapshot_health())
            return
        if self.path == "/v1/models":
            self._json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": model_id,
                            "object": "model",
                            "created": 1710000000,
                            "owned_by": "local",
                            "use_case": (
                                "hq_offline"
                                if model_id in {MODEL_ID, "s2-pro-mlx"}
                                else "compat"
                            ),
                        }
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
        fish = self.server.fish  # type: ignore[attr-defined]
        # B1-T4: track in-flight synthesis depth (counter mutation is GIL-unsafe
        # for ``+=`` on a bare int, so we go through the dedicated lock on
        # FishMLXServer). Decrement always runs in ``finally`` so exceptions
        # don't strand the counter and stall back-pressure.
        fish._inc_in_flight()
        try:
            audio, media_type, timing = fish.synthesize(self._read_json())
            self._bytes(HTTPStatus.OK, audio, media_type, timing, queue_depth=fish._snapshot_in_flight())
        except Exception as exc:
            # B8-T1: surface the failure in /health.last_error so a stuck or
            # repeatedly-failing instance is not silent from the health
            # endpoint.
            fish._record_error(exc)
            self._json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": str(exc), "type": exc.__class__.__name__}},
                queue_depth=fish._snapshot_in_flight(),
            )
        finally:
            fish._dec_in_flight()

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8881)
    parser.add_argument(
        "--model-path",
        default="checkpoints/fish-audio-s2-pro-8bit-mlx-normalized",
        help="Local path to the downloaded MLX model snapshot.",
    )
    parser.add_argument(
        "--lazy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Lazy-load weights (default: eager for low first-request latency).",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a short synthesis after model load to prime MLX caches.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.fish = FishMLXServer(  # type: ignore[attr-defined]
        model_path=model_path,
        lazy=args.lazy,
        warmup=args.warmup,
    )
    if not args.lazy:
        httpd.fish.ensure_model()  # type: ignore[attr-defined]

    print(f"Fish MLX host server listening on http://{args.host}:{args.port}", flush=True)
    print(f"Model path: {model_path}", flush=True)
    print(
        "Fish S2 MLX is HQ/offline only; use VibeVoice/Chatterbox for live conversation.",
        flush=True,
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()
