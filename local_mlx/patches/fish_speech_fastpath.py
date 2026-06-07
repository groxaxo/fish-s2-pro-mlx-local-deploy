"""Runtime patches for mlx-audio Fish Speech generation hot path.

Optimizations (backed by MLX Context7 docs):
1. Reuse fast KV-cache across semantic steps (trim+reuse vs realloc 131x)
2. Pre-allocated codebook buffer replaces mx.concatenate (fusion barrier removal)
3. Deferred evaluation: single mx.eval per outer-loop step, not per sub-operation
4. Compiled greedy sampling via @mx.compile
5. Minimize .item() forced-sync points (Context7: "scalar access forces eval")

Accurate profiling: When FISH_MLX_PROFILE=1, explicit mx.eval() barriers are
inserted at each phase boundary so wall-clock timers measure actual GPU/Metal
execution, not just lazy graph construction.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import mlx.core as mx

_PROFILE_ENABLED = os.environ.get("FISH_MLX_PROFILE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

if TYPE_CHECKING:
    from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model

_PATCHED = False
_PROFILE_LAST: dict[str, float] = {}


@mx.compile
def _fast_sample_greedy(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1).astype(mx.int32)


def _sample_semantic_fast(
    self: "Model",
    logits: mx.array,
    previous_semantic_tokens: mx.array,
    top_p: float,
    top_k: int,
    temperature: float,
) -> mx.array:
    from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import (
        RAS_HIGH_TEMP,
        RAS_HIGH_TOP_P,
        _sample_logits,
    )

    if self.semantic_logit_bias is None:
        raise ValueError("Semantic logits bias is not initialized.")

    biased_logits = logits + self.semantic_logit_bias.astype(logits.dtype)
    normal = _sample_logits(
        biased_logits, temperature=temperature, top_p=top_p, top_k=top_k
    )

    if previous_semantic_tokens.size == 0:
        return normal

    high_temp = _sample_logits(
        biased_logits,
        temperature=RAS_HIGH_TEMP,
        top_p=RAS_HIGH_TOP_P,
        top_k=top_k,
    )

    token = normal.reshape(-1)[0]
    prev = previous_semantic_tokens.reshape(-1)
    in_window = mx.any(prev == token)
    is_semantic = (token >= self.config.semantic_start_token_id) & (
        token <= self.config.semantic_end_token_id
    )
    use_high = in_window & is_semantic
    chosen = mx.where(use_high, high_temp.reshape(-1)[0], token)
    return chosen.reshape(normal.shape).astype(normal.dtype)


def _generate_codes_for_batch_patched(
    self: "Model",
    conversation,
    batch_text: str,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
) -> mx.array:
    from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import (
        RAS_WIN_SIZE,
        Conversation,
        Message,
    )
    from mlx_audio.tts.models.fish_qwen3_omni.prompt import TextPart
    from mlx_audio.tts.models.fish_qwen3_omni.tokenizer import IM_END_TOKEN
    from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import _sample_logits

    if self.tokenizer is None:
        raise ValueError("Tokenizer not loaded. Call post_load_hook first.")

    t0 = time.perf_counter() if _PROFILE_ENABLED else 0.0
    slow_ms = fast_ms = sample_ms = eval_ms = 0.0

    prompt_conversation = Conversation(list(conversation.messages))
    prompt_conversation.append(
        Message(
            role="assistant",
            parts=[],
            modality="voice",
            add_im_start=True,
            add_im_end=False,
        )
    )
    prompt = prompt_conversation.encode_for_inference(
        self.tokenizer, num_codebooks=self.model.num_codebooks
    )
    prompt = prompt[None, :, :]

    cache = self.model.make_cache()
    result = self.model(prompt, cache=cache)
    logits = result.logits[:, -1]
    hidden_state = result.hidden_states[:, -1]

    previous_semantic_tokens: list[int] = []
    generated_steps = []
    im_end_id = self.tokenizer.get_token_id(IM_END_TOKEN)
    text_token_count = len(self.tokenizer.encode(batch_text))
    semantic_token_budget = min(max_new_tokens, max(32, text_token_count * 12))
    num_cb = self.model.num_codebooks
    greedy = temperature <= 0

    # Allocate fast cache once; reuse across all semantic steps via trim()
    fast_cache = self.model.make_fast_cache()

    for step in range(semantic_token_budget):
        # Phase 1: Sample semantic token (includes pending graph evaluation)
        if _PROFILE_ENABLED:
            ts = time.perf_counter()

        ras_window = (
            mx.array(previous_semantic_tokens, dtype=mx.int32)
            if previous_semantic_tokens
            else mx.array([], dtype=mx.int32)
        )
        semantic_token = _sample_semantic_fast(
            self,
            logits=logits,
            previous_semantic_tokens=ras_window,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        # .item() forces eval of pending lazy graph from previous step
        semantic_token_id = int(semantic_token[0].item())

        if _PROFILE_ENABLED:
            sample_ms += time.perf_counter() - ts

        if semantic_token_id == im_end_id:
            break

        previous_semantic_tokens.append(semantic_token_id)
        previous_semantic_tokens = previous_semantic_tokens[-RAS_WIN_SIZE:]

        semantic_code = (
            semantic_token - self.config.semantic_start_token_id
        ).astype(mx.int32)
        semantic_code = mx.clip(
            semantic_code, 0, self.config.audio_decoder_config.vocab_size - 1
        )

        # Phase 2: Fast AR -- prefill + 9 residual codebook steps
        if _PROFILE_ENABLED:
            tf = time.perf_counter()

        for c in fast_cache:
            c.trim(c.offset)

        self.model.fast_forward_cached(hidden_state, fast_cache)
        fast_hidden = self.model.fast_embeddings(semantic_code)

        codebook_buf = mx.zeros((1, num_cb), dtype=mx.int32)
        codebook_buf[:, 0] = semantic_code

        for i in range(1, num_cb):
            residual_logits = self.model.fast_forward_cached(fast_hidden, fast_cache)
            if greedy:
                residual_token = _fast_sample_greedy(residual_logits)
            else:
                residual_token = _sample_logits(
                    residual_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            codebook_buf[:, i] = residual_token
            fast_hidden = self.model.fast_embeddings(residual_token)

        codebook_row = codebook_buf[0]

        if _PROFILE_ENABLED:
            # Force eval to measure actual fast AR execution time
            mx.eval(codebook_row)
            fast_ms += time.perf_counter() - tf

        generated_steps.append(codebook_row)

        # Phase 3: Build next input + eval + slow model step
        next_input = mx.concatenate(
            [
                semantic_token[:, None].astype(mx.int32),
                codebook_row[None, :],
            ],
            axis=1,
        )

        if _PROFILE_ENABLED:
            ts = time.perf_counter()
            mx.eval(next_input)
            eval_ms += time.perf_counter() - ts
            ts = time.perf_counter()

        next_result = self.model(next_input[:, :, None], cache=cache)
        logits = next_result.logits[:, -1]
        hidden_state = next_result.hidden_states[:, -1]

        if _PROFILE_ENABLED:
            # Force eval to measure actual slow model execution time
            mx.eval(logits, hidden_state)
            slow_ms += time.perf_counter() - ts

        if not _PROFILE_ENABLED:
            # In production (no profiling), use single eval boundary for throughput
            mx.eval(next_input)

    if not generated_steps:
        raise RuntimeError(
            f"No audio tokens were generated for batch text: {batch_text!r}"
        )

    global _PROFILE_LAST
    if _PROFILE_ENABLED:
        _PROFILE_LAST = {
            "total_ms": (time.perf_counter() - t0) * 1000.0,
            "slow_ms": slow_ms * 1000.0,
            "fast_ms": fast_ms * 1000.0,
            "sample_ms": sample_ms * 1000.0,
            "eval_ms": eval_ms * 1000.0,
            "semantic_steps": float(len(generated_steps)),
        }

    return mx.stack(generated_steps, axis=1).astype(mx.int32)


def _generate_codes_for_text_batch_patched(
    self: "Model",
    conversations: list,
    batch_texts: list[str],
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
) -> list[mx.array]:
    from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import (
        RAS_WIN_SIZE,
        _sample_logits,
    )
    from mlx_audio.tts.models.fish_qwen3_omni.tokenizer import IM_END_TOKEN

    if self.tokenizer is None:
        raise ValueError("Tokenizer not loaded. Call post_load_hook first.")

    batch_size = len(conversations)
    if batch_size == 0:
        return []

    prompt, attention_mask = self._prepare_batched_prompt_inputs(conversations)
    cache = self.model.make_cache()
    result = self.model(prompt, cache=cache, attention_mask=attention_mask)
    logits = result.logits[:, -1]
    hidden_state = result.hidden_states[:, -1]

    previous_semantic_tokens: list[list[int]] = [[] for _ in range(batch_size)]
    generated_steps: list[list[mx.array]] = [[] for _ in range(batch_size)]
    finished = [False] * batch_size
    im_end_id = self.tokenizer.get_token_id(IM_END_TOKEN)
    token_budgets = [
        min(max_new_tokens, max(32, len(self.tokenizer.encode(text)) * 12))
        for text in batch_texts
    ]
    max_budget = max(token_budgets)
    im_end_tokens = mx.full((batch_size,), im_end_id, dtype=mx.int32)
    greedy = temperature <= 0
    num_cb = self.model.num_codebooks

    # Allocate fast cache once; reuse across all steps via trim()
    fast_cache = self.model.make_fast_cache()

    for step in range(max_budget):
        active = [
            (not finished[idx]) and step < token_budgets[idx]
            for idx in range(batch_size)
        ]
        if not any(active):
            break

        sampled_semantic = self._sample_semantic_batch(
            logits=logits,
            previous_semantic_tokens=previous_semantic_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        active_mask = mx.array(active, dtype=mx.bool_)
        semantic_token = mx.where(active_mask, sampled_semantic, im_end_tokens)

        semantic_token_ids = semantic_token.tolist()
        if not isinstance(semantic_token_ids, list):
            semantic_token_ids = [semantic_token_ids]
        should_continue = []
        for idx, token_id in enumerate(semantic_token_ids):
            keep_generating = active[idx] and token_id != im_end_id
            should_continue.append(keep_generating)
            if active[idx] and not keep_generating:
                finished[idx] = True

        if not any(should_continue):
            break

        continue_mask = mx.array(should_continue, dtype=mx.bool_)
        semantic_code = (
            semantic_token - self.config.semantic_start_token_id
        ).astype(mx.int32)
        semantic_code = mx.clip(
            semantic_code, 0, self.config.audio_decoder_config.vocab_size - 1
        )
        semantic_code = mx.where(
            continue_mask, semantic_code, mx.zeros_like(semantic_code)
        )

        # Trim fast cache back to offset 0 for this step
        for c in fast_cache:
            c.trim(c.offset)

        self.model.fast_forward_cached(hidden_state, fast_cache)
        fast_hidden = self.model.fast_embeddings(semantic_code)

        # Pre-allocated codebook buffer instead of incremental concatenate
        codebook_buf = mx.zeros((batch_size, num_cb), dtype=mx.int32)
        codebook_buf[:, 0] = semantic_code

        for i in range(1, num_cb):
            residual_logits = self.model.fast_forward_cached(fast_hidden, fast_cache)
            if greedy:
                residual_token = _fast_sample_greedy(residual_logits)
            else:
                residual_token = _sample_logits(
                    residual_logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            residual_token = mx.where(
                continue_mask, residual_token, mx.zeros_like(residual_token)
            )
            codebook_buf[:, i] = residual_token
            fast_hidden = self.model.fast_embeddings(residual_token)

        for idx, keep_generating in enumerate(should_continue):
            if not keep_generating:
                continue
            token_id = semantic_token_ids[idx]
            previous_semantic_tokens[idx].append(token_id)
            previous_semantic_tokens[idx] = previous_semantic_tokens[idx][
                -RAS_WIN_SIZE:
            ]
            generated_steps[idx].append(codebook_buf[idx])
            if step + 1 >= token_budgets[idx]:
                finished[idx] = True

        next_input = mx.concatenate(
            [semantic_token[:, None].astype(mx.int32), codebook_buf], axis=1
        )
        attention_mask = mx.concatenate(
            [attention_mask, mx.ones((batch_size, 1), dtype=attention_mask.dtype)],
            axis=1,
        )

        mx.eval(next_input)

        next_result = self.model(
            next_input[:, :, None],
            cache=cache,
            attention_mask=attention_mask,
        )
        logits = next_result.logits[:, -1]
        hidden_state = next_result.hidden_states[:, -1]

        if all(finished):
            break
        if step > 0 and step % 50 == 0:
            mx.clear_cache()

    empty_indices = [
        idx
        for idx, sequence_steps in enumerate(generated_steps)
        if not sequence_steps
    ]
    if empty_indices:
        raise RuntimeError(
            "No audio tokens were generated for batch sequence(s): "
            + ", ".join(str(idx) for idx in empty_indices)
        )

    return [
        mx.stack(sequence_steps, axis=1).astype(mx.int32)
        for sequence_steps in generated_steps
    ]


def apply_fish_speech_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from mlx_audio.tts.models.fish_qwen3_omni import fish_speech as fs

    fs.Model._generate_codes_for_batch = _generate_codes_for_batch_patched
    fs.Model._generate_codes_for_text_batch = _generate_codes_for_text_batch_patched
    _PATCHED = True


def get_profile_stats() -> dict[str, float]:
    return dict(_PROFILE_LAST)
