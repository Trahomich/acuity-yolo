"""JSON-логирование в stderr (как в acuity-server: slog JSON → stderr).

Простой stdlib-форматтер: одна строка JSON на событие с полями message,
level, time и всеми extra-полями. Уровень из ACUITY_LOG_LEVEL / settings.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

# Имена полей выровнены с тем, как их ждёт стеклянка acuity-server
# (component/level/msg/err/time) для единообразия логов.
_RESERVED = {"message", "level", "time"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                    + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "msg": record.getMessage(),
        }
        # extra-поля из logger.info(..., extra={...}) попадают в __dict__.
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_") or key in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "module", "msecs",
                "message", "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName", "taskName",
            }:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        if record.exc_info:
            payload["err"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "info") -> None:
    """Конфигурирует root-logger с JSON-выводом в stderr."""
    root = logging.getLogger()
    # Чистим хендлеры (uvicorn может навесить свои при reload).
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(_parse_level(level))
    # uvicorn-логи — в тот же формат/поток.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(noisy)
        lg.handlers = []
        lg.propagate = True


def _parse_level(level: str) -> int:
    table = {"debug": logging.DEBUG, "info": logging.INFO,
             "warn": logging.WARNING, "warning": logging.WARNING,
             "error": logging.ERROR}
    return table.get(level.strip().lower(), logging.INFO)
