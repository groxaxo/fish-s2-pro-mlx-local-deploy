#!/usr/bin/env python3
"""Normalize cs2764 Fish S2 Pro MLX weights for mlx-audio's Fish S2 branch.

The cs2764 snapshot stores text/audio decoder keys without the `model.` prefix
that `lucasnewman/mlx-audio@fish-audio-s2` expects. This rewrites only
`model.safetensors`; all tokenizer/config/codec files are copied as-is.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx


COPY_FILES = (
    ".gitattributes",
    "LICENSE.md",
    "README.md",
    "chat_template.jinja",
    "codec.safetensors",
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    dest = args.dest.expanduser().resolve()
    if not (source / "model.safetensors").exists():
        raise SystemExit(f"Missing source model.safetensors: {source}")

    dest.mkdir(parents=True, exist_ok=True)
    for name in COPY_FILES:
        src = source / name
        if src.exists():
            shutil.copy2(src, dest / name)

    weights = mx.load(str(source / "model.safetensors"))
    normalized = {}
    for key, value in weights.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        if key.startswith("fast_"):
            new_key = f"audio_decoder.{key[len('fast_') :]}"
        elif key.startswith("codebook_embeddings."):
            new_key = f"audio_decoder.{key}"
        else:
            new_key = f"text_model.model.{key}"
        normalized[new_key] = value
    mx.save_safetensors(str(dest / "model.safetensors"), normalized)

    index = {
        "metadata": {
            "total_parameters": int(sum(value.size for value in normalized.values())),
        },
        "weight_map": {key: "model.safetensors" for key in sorted(normalized)},
    }
    (dest / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote normalized checkpoint to {dest}")


if __name__ == "__main__":
    main()
