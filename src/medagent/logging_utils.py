from __future__ import annotations

import logging
import time
from typing import Any, Dict

try:  # Prefer orjson if available
    import orjson as _json
except Exception:  # pragma: no cover
    import json as _json  # type: ignore


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "time": getattr(record, "asctime", None) or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        for k in ("request_id", "path", "method", "status_code", "latency_ms"):
            v = getattr(record, k, None)
            if v is not None:
                payload[k] = v
        try:
            return _json.dumps(payload).decode() if hasattr(_json, "dumps") else _json.dumps(payload)  # type: ignore[attr-defined]
        except Exception:
            return str(payload)


def get_logger(name: str = "medagent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(JsonFormatter())
        logger.addHandler(h)
    return logger

