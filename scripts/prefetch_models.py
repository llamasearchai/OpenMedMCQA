#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

from huggingface_hub import snapshot_download


def prefetch(model_ids: List[str], cache_dir: str | None = None) -> None:
    cache_dir = cache_dir or os.environ.get("HF_HOME") or str(Path(".hf-cache").absolute())
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    for mid in model_ids:
        print(f"Prefetching {mid} into {cache_dir} ...", flush=True)
        snapshot_download(repo_id=mid, cache_dir=cache_dir, local_files_only=False, ignore_regex=[r".*onnx.*", r".*tf.*", r".*safetensors.index.*"])  # noqa: E501
    print("Done.")


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python scripts/prefetch_models.py <model> [<model> ...]", file=sys.stderr)
        print("Example: python scripts/prefetch_models.py allenai/scibert_scivocab_uncased", file=sys.stderr)
        return 2
    prefetch(argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

