#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from typing import List


def main(argv: List[str]) -> int:
    try:
        from datasets import load_dataset  # lazy import
    except Exception as e:  # pragma: no cover
        print("Install optional extra: pip install '.[data]'", file=sys.stderr)
        print(f"datasets import error: {e}", file=sys.stderr)
        return 2

    if len(argv) < 2:
        print("Usage: python scripts/prefetch_datasets.py <dataset_name_or_path> [<config>]", file=sys.stderr)
        print("Example: python scripts/prefetch_datasets.py medmcqa", file=sys.stderr)
        return 2

    name = argv[1]
    config = argv[2] if len(argv) > 2 else None
    print(f"Prefetching dataset: {name} config={config!r}")
    load_dataset(name, name=config, split=None)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

