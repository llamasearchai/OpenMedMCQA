from __future__ import annotations

import os
import subprocess
import webbrowser
from .config import settings


def serve_datasette(open_browser: bool = True, port: int = 8080) -> None:
    cmd = ["datasette", settings.db_path, "--setting", "sql_time_limit_ms", "30000", "--port", str(port)]
    meta = os.path.abspath("datasette_metadata.json")
    if os.path.exists(meta):
        cmd += ["--metadata", meta]
    p = subprocess.Popen(cmd)
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{port}")
    p.wait()

