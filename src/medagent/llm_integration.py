from __future__ import annotations

import shutil
import subprocess
from typing import Optional


def llm_available() -> bool:
    return shutil.which("llm") is not None


def run_llm_cmd(prompt: str, model: str = "openai:gpt-4o-mini", system: Optional[str] = None) -> str:
    if not llm_available():
        raise RuntimeError("The `llm` CLI is not installed. See https://llm.datasette.io/")
    cmd = ["llm", "-m", model]
    if system:
        cmd += ["--system", system]
    cmd += ["--", prompt]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return res.stdout.strip()


