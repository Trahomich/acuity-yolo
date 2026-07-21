"""JSON-логирование в stderr + опциональная отправка в VictoriaLogs (OTLP HTTP).

Простой stdlib-форматтер: одна строка JSON на событие с полями message,
level, time и всеми extra-полями. Уровень из ACUITY_LOG_LEVEL / settings.

Если задан victorialogs_url — логи дублируются в VictoriaMetrics/VictoriaLogs
через OTLP HTTP API (/insert/opentelemetry/v1/logs). Отправка идёт в фоновом
потоке через очередь (drop+warn при переполнении / недоступности).
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

# Имена полей выровнены с тем, как их ждёт стеклянка acuity-server
# (component/level/msg/err/time) для единообразия логов.
_RESERVED = {"message", "level", "time"}

# Внутренние поля LogRecord, которые не идут в payload (не extra).
_LOGRECORD_BUILTIN = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "module", "msecs",
    "message", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
})

# Map Python log level → OTLP severity number (см. OTLP logs data model).
# VictoriaLogs принимает любое значение, но для корректной фильтрации по
# severity ставим канонические значения.
_SEVERITY_NUM = {
    "DEBUG": 5, "INFO": 9, "WARNING": 13, "ERROR": 17, "CRITICAL": 21,
}


def _record_to_payload(record: logging.LogRecord) -> dict[str, Any]:
    """Достаёт из LogRecord единый payload (msg/level/time + extra-поля)."""
    payload: dict[str, Any] = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                + f".{int(record.msecs):03d}Z",
        "level": record.levelname.lower(),
        "msg": record.getMessage(),
    }
    for key, value in record.__dict__.items():
        if key in _RESERVED or key.startswith("_") or key in _LOGRECORD_BUILTIN:
            continue
        try:
            json.dumps(value)
            payload[key] = value
        except (TypeError, ValueError):
            payload[key] = repr(value)
    if record.exc_info:
        payload["err"] = logging.Formatter().formatException(record.exc_info)
    return payload


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(_record_to_payload(record), ensure_ascii=False)


class VictoriaLogsHandler(logging.Handler):
    """Асинхронная отправка логов в VictoriaLogs через OTLP HTTP.

    Потокобезопасен. emit() не блокируется — кладёт запись в очередь.
    Фоновый поток забирает батчами (batch_size либо раз в flush_interval)
    и шлёт POST. При недоступности/переполнении — drop + warn в stderr
    (раз в 60с, чтобы не залогировать саму ошибку отправки в цикле).
    """

    def __init__(
        self,
        url: str,
        service_name: str = "acuity-yolo",
        flush_interval: float = 2.0,
        batch_size: int = 64,
        queue_max: int = 4096,
    ) -> None:
        super().__init__()
        self.url = url
        self.service_name = service_name
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self._dropped = 0
        self._last_warn = 0.0
        # urllib с таймаутом, чтобы поток не завис при недоступности.
        self._thread = threading.Thread(
            target=self._run, name="victorialogs-sender", daemon=True
        )
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        payload = _record_to_payload(record)
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            # Очередь переполнена — дропаем remote-копию. stderr-вывод уже
            # сделал JsonFormatter; здесь только считаем для warn'а.
            self._dropped += 1
            self._maybe_warn_dropped()

    def _maybe_warn_dropped(self) -> None:
        """Раз в 60с пишет в stderr предупреждение о числе дропов."""
        now = time.time()
        if now - self._last_warn < 60.0 or self._dropped == 0:
            return
        self._last_warn = now
        n = self._dropped
        self._dropped = 0
        sys.stderr.write(
            f"[victorialogs] dropped {n} log records (queue full or backend down)\n"
        )
        sys.stderr.flush()

    def _build_otlp_payload(self, batch: list[dict[str, Any]]) -> bytes:
        """Формирует OTLP HTTP JSON body из батча лог-записей.

        Формат: resourceLogs → scopeLogs → logRecords. Каждая logRecord:
          timeUnixNano, observedTimeUnixNano, severityNumber, severityText,
          body{stringValue}, attributes[{key,value{stringValue}}].
        Extra-поля идут как attributes (фильтрация/группировка в VictoriaLogs).
        """
        now_ns = str(int(time.time() * 1e9))
        log_records = []
        for p in batch:
            # Парсим ISO-время записи обратно в ns.
            ts_str = p.get("time", "")
            ts_unix_ns = now_ns
            try:
                # "2026-07-19T..." → unix-секунды.
                dt = time.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S")
                ts_unix_ns = str(int(time.mktime(dt)) * 10**9)
            except (ValueError, OverflowError):
                pass
            level_upper = str(p.get("level", "info")).upper()
            sev_num = _SEVERITY_NUM.get(level_upper, 9)
            attrs = [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in p.items()
                if k not in ("msg", "time", "level")
            ]
            log_records.append({
                "timeUnixNano": ts_unix_ns,
                "observedTimeUnixNano": now_ns,
                "severityNumber": sev_num,
                "severityText": level_upper,
                "body": {"stringValue": str(p.get("msg", ""))},
                "attributes": attrs,
            })
        return json.dumps({
            "resourceLogs": [{
                "resource": {
                    "attributes": [{
                        "key": "service.name",
                        "value": {"stringValue": self.service_name},
                    }]
                },
                "scopeLogs": [{
                    "scope": {"name": "acuity-yolo"},
                    "logRecords": log_records,
                }],
            }],
        }).encode("utf-8")

    def _post(self, body: bytes) -> bool:
        """POST батча в OTLP endpoint. Возвращает True при успехе."""
        req = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._dropped += 1
            self._maybe_warn_dropped()
            sys.stderr.write(f"[victorialogs] POST failed: {exc}\n")
            sys.stderr.flush()
            return False

    def _run(self) -> None:
        """Главный цикл фонового потока: копит и флешит батчи."""
        batch: list[dict[str, Any]] = []
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=self.flush_interval)
            except queue.Empty:
                if batch:
                    self._flush(batch)
                    batch = []
                continue
            if item is None:  # sentinel для остановки
                break
            batch.append(item)
            if len(batch) >= self.batch_size:
                self._flush(batch)
                batch = []
        # Финальный flush при остановке.
        if batch:
            self._flush(batch)

    def _flush(self, batch: list[dict[str, Any]]) -> None:
        body = self._build_otlp_payload(batch)
        if not self._post(body):
            # При неудаче не ретраим (drop+warn стратегия) — журнал TTL/ingest
            # на стороне VictoriaLogs примет следующий батч.
            pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)  # разблокировать get(timeout=...)
        except queue.Full:
            pass
        self._thread.join(timeout=self.flush_interval + 5)
        super().close()


def setup_logging(
    level: str = "info",
    victorialogs_url: str = "",
    service_name: str = "acuity-yolo",
    flush_interval: float = 2.0,
    batch_size: int = 64,
    queue_max: int = 4096,
) -> None:
    """Конфигурирует root-logger: stderr (JSON) + опционально VictoriaLogs.

    Если victorialogs_url непустой — добавляет асинхронный remote-handler.
    """
    root = logging.getLogger()
    # Чистим хендлеры (uvicorn может навесить свои при reload).
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    if victorialogs_url:
        root.addHandler(VictoriaLogsHandler(
            url=victorialogs_url,
            service_name=service_name,
            flush_interval=flush_interval,
            batch_size=batch_size,
            queue_max=queue_max,
        ))
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
