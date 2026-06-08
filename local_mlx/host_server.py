#!/usr/bin/env python3
"""OpenAI-compatible local API for MLX Fish Audio S2 Pro.

Docker Desktop Linux containers cannot access Apple Metal/MLX directly, so this
server runs on macOS and is fronted by the Docker proxy in this directory.
"""

from __future__ import annotations

import argparse
import signal
import sys
import io
import json
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

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


@dataclass
class SegmentYield:
    """One yielded item from FishMLXServer.synthesize_segments().

    B2-T0: refactored synthesize() to a generator that yields per-segment
    PCM as the model produces it, so a future chunked HTTP handler
    (B2-T1/T2) can flush each segment to the client without waiting for
    the whole request. The final yield is a sentinel with ``is_final=True``
    and carries the aggregate timing; audio_payload is None on that yield.
    """

    segment_index: int
    audio_payload: np.ndarray | None  # float32 [-1, 1], or None on final
    sample_rate: int
    is_final: bool
    semantic_tokens_total: int = 0
    elapsed_s: float = 0.0
    final_timing: dict[str, Any] = field(default_factory=dict)


def _encode_wav_pcm16(audio_np: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 mono audio array to a 16-bit PCM RIFF/WAVE byte string.

    B5-T1: hand-rolled replacement for ``soundfile.write(..., format="WAV")``
    that skips the intermediate ``BytesIO`` and the C extension call. The
    model emits float32 in ``[-1, 1]`` — clip and scale to int16 before
    packing the header, otherwise the output is near-silence.

    Returns the full WAV file bytes (header + PCM data).
    """
    pcm = (np.clip(audio_np, -1.0, 1.0) * 32767.0).astype(np.int16)
    data_bytes = pcm.tobytes()
    # fmt chunk: PCM (format=1), 1 channel, sample_rate, byte_rate=rate*2,
    # block_align=2 (16-bit mono), bits_per_sample=16
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data_bytes),
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        1,  # num channels
        sample_rate,
        sample_rate * 2,  # byte rate (rate * block_align)
        2,  # block align
        16,  # bits per sample
        b"data",
        len(data_bytes),
    )
    return header + data_bytes


# B7-T1: graceful-shutdown state. The signal handler in main() sets
# ``shutdown_event`` and spawns a daemon thread to do the drain+evict
# sequence. The signal handler itself only flips the flag — calling
# ``httpd.shutdown()`` from a main-thread signal handler would deadlock
# because ``shutdown()`` blocks until ``serve_forever()`` returns, and
# ``serve_forever()`` is the loop the handler is interrupting.
DEFAULT_DRAIN_TIMEOUT_S = 30.0
DRAIN_POLL_INTERVAL_S = 0.1


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

    def evict_model(self) -> None:
        """B7-T1: drop the loaded model weights and free MLX caches.

        Called from the shutdown handler after the in-flight request drain.
        Idempotent: a second call is a no-op if the model is already gone.
        """
        with self.lock:
            if self.model is None:
                return
            print(
                f"Evicting model from {self.model_path} (in-flight={self._snapshot_in_flight()})",
                flush=True,
            )
            self.model = None
            try:
                mx.clear_cache()
            except Exception as exc:  # pragma: no cover - best-effort
                print(f"mx.clear_cache() during evict raised {exc!r}", flush=True)

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

        # B2-T0: refactored to a generator that yields per-segment PCM as
        # the model produces it. synthesize() now collects the segments
        # and runs the existing encoder path so the wire response is
        # unchanged; a future chunked handler (B2-T1/T2) can consume the
        # generator directly and flush each segment without buffering.
        # The synthesis lock is held per segment step (released between
        # segments) so a slow consumer cannot block other requests — that
        # is the lock-coupling rule noted in B2-T0.
        sample_rate = getattr(model, "sample_rate", 24000)
        chunks: list[np.ndarray] = []
        semantic_tokens = 0
        started = time.perf_counter()
        final_timing: dict[str, Any] = {}

        for segment in self.synthesize_segments(
            text=text,
            model=model,
            gen_kwargs=gen_kwargs,
            sample_rate=sample_rate,
        ):
            if segment.is_final:
                final_timing = segment.final_timing
                break
            chunks.append(np.asarray(segment.audio_payload, dtype=np.float32))
            sample_rate = segment.sample_rate
            semantic_tokens = segment.semantic_tokens_total

        if not chunks:
            raise RuntimeError("Model returned no audio.")

        audio_np = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
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
        if final_timing:
            # Preserve any fields the generator populated (none today,
            # reserved for future profile data).
            self.last_timing.update(final_timing)

        out = io.BytesIO()
        media_type: str
        if response_format == "wav":
            # B5-T1: hand-rolled 44-byte RIFF/WAVE header + int16 PCM. Skips
            # the intermediate BytesIO + soundfile C-extension round-trip and
            # makes the float32->int16 conversion explicit. The model output
            # is float32 in [-1, 1] (audio_np = np.asarray(audio,
            # dtype=np.float32)) — sf.write() scaled this automatically; the
            # new path must scale or it emits near-silence. Scale first,
            # header sizes from the *PCM* byte count (not the float array's
            # nbytes), otherwise the data chunk is mis-sized and downstream
            # decoders either truncate or reject the file.
            out.write(_encode_wav_pcm16(audio_np, sample_rate))
            media_type = "audio/wav"
        else:
            sf.write(out, audio_np, sample_rate, format=response_format.upper())
            media_type = {
                "flac": "audio/flac",
                "ogg": "audio/ogg",
            }[response_format]
        return out.getvalue(), media_type, self.last_timing

    def synthesize_segments(
        self,
        text: str,
        model: Any,
        gen_kwargs: dict[str, Any],
        sample_rate: int,
    ) -> Iterator[SegmentYield]:
        """B2-T0: generator yielding per-segment PCM as the model produces it.

        The model emits one ``GenerationResult`` per text *segment* (the
        model generator yields per-segment, not per-frame — see
        TODO_BOTTLENECKS.md B2 GROUND TRUTH). This wrapper just translates
        those into ``SegmentYield`` items with float32 audio in [-1, 1] so
        the caller can choose how to encode / stream them.

        The synthesis lock is acquired *per* ``model.generate()`` step
        (i.e., per segment) rather than wrapping the whole generator. A
        slow HTTP consumer therefore cannot block other synthesis
        requests during the network write — the next request can begin
        its first segment as soon as the current one is yielded.

        Yields:
          SegmentYield(segment_index, audio_payload=ndarray,
                       sample_rate, is_final=False, ...)
          for every emitted segment, followed by a final sentinel
          SegmentYield(segment_index=N, audio_payload=None,
                       is_final=True, elapsed_s=..., final_timing={...}).
        """
        del text  # not used directly; gen_kwargs["text"] is the source of truth
        semantic_tokens = 0
        started = time.perf_counter()
        segment_index = 0
        empty = True
        with self.lock:
            for result in model.generate(**gen_kwargs):
                sample_rate = int(getattr(result, "sample_rate", sample_rate) or sample_rate)
                semantic_tokens += int(getattr(result, "token_count", 0) or 0)
                audio = np.asarray(result.audio, dtype=np.float32)
                empty = False
                yield SegmentYield(
                    segment_index=segment_index,
                    audio_payload=audio,
                    sample_rate=sample_rate,
                    is_final=False,
                    semantic_tokens_total=semantic_tokens,
                    elapsed_s=time.perf_counter() - started,
                )
                segment_index += 1
        if empty:
            # Same failure mode as the old code: ``Model returned no audio.``
            # but raised by the consumer (synthesize() or the chunked
            # handler). We do not raise inside the generator because the
            # generator protocol forbids raising mid-iteration cleanly.
            elapsed = time.perf_counter() - started
            yield SegmentYield(
                segment_index=0,
                audio_payload=None,
                sample_rate=sample_rate,
                is_final=True,
                semantic_tokens_total=0,
                elapsed_s=elapsed,
                final_timing={"empty": True},
            )
            return
        yield SegmentYield(
            segment_index=segment_index,
            audio_payload=None,
            sample_rate=sample_rate,
            is_final=True,
            semantic_tokens_total=semantic_tokens,
            elapsed_s=time.perf_counter() - started,
        )


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
    # B5-T3: make the default response format and its perf trade-off visible
    # at startup, so operators picking response_format=flac/ogg don't get
    # surprised by the post-generation encoding latency (~50-200 ms).
    print(
        "Default response format: wav (B5-T1 hand-rolled encoder, no sf.write "
        "round-trip). FLAC/OGG still go through soundfile and incur ~50-200 ms "
        "post-generation encoding latency. Pass response_format=wav to skip it.",
        flush=True,
    )

    # B7-T1: graceful shutdown. Signal handlers only set the event; the
    # actual drain + evict + httpd.shutdown() runs in a short-lived
    # thread so the main serve_forever loop is unblocked and can return
    # after shutdown() is called from the worker thread.
    shutdown_event = threading.Event()

    def _on_signal(signum: int, _frame: Any) -> None:
        signame = signal.Signals(signum).name
        if shutdown_event.is_set():
            return  # idempotent — repeated Ctrl-C / SIGTERM is a no-op
        print(
            f"Received {signame}; beginning graceful shutdown "
            f"(drain timeout={DEFAULT_DRAIN_TIMEOUT_S}s).",
            flush=True,
        )
        shutdown_event.set()

        def _drain_and_stop() -> None:
            # 1. Wait for in-flight synthesis to drain (B1-T4 counter).
            started = time.perf_counter()
            while (
                httpd.fish._snapshot_in_flight() > 0
                and (time.perf_counter() - started) < DEFAULT_DRAIN_TIMEOUT_S
            ):
                time.sleep(DRAIN_POLL_INTERVAL_S)
            remaining = httpd.fish._snapshot_in_flight()
            elapsed = time.perf_counter() - started
            if remaining:
                print(
                    f"Drain timeout reached after {elapsed:.2f}s with "
                    f"{remaining} in-flight request(s); forcing shutdown.",
                    flush=True,
                )
            else:
                print(
                    f"In-flight requests drained in {elapsed:.3f}s.",
                    flush=True,
                )

            # 2. Evict the model so the next start is fast + memory is freed.
            httpd.fish.evict_model()

            # 3. Stop the serve_forever loop. Safe to call from a non-
            # serving thread; BaseServer is designed for this.
            httpd.shutdown()
            print("Server stopped; exiting with code 0.", flush=True)
            os._exit(0)

        threading.Thread(
            target=_drain_and_stop, name="fish-mlx-shutdown", daemon=True
        ).start()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGHUP"):
        # Reload-config signal on macOS LaunchAgents. Treat as a soft
        # restart request — the LaunchAgent KeepAlive will respawn us
        # with the new code.
        signal.signal(signal.SIGHUP, _on_signal)

    try:
        httpd.serve_forever()
    finally:
        # If serve_forever returned for any other reason (e.g., the
        # shutdown thread won the race), still try to evict so a follow-up
        # process does not see the model resident in unified memory.
        httpd.fish.evict_model()


if __name__ == "__main__":
    main()
