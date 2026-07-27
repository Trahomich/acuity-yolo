"""JSON-логирование в stderr + опциональная отправка в VictoriaLogs (OTLP HTTP).

Простой stdlib-форматтер: одна строка JSON на событие с полями message,
level, time и всеми extra-полями. Уровень из ACUITY_LOG_LEVEL / settings.

Если задан victorialogs_url — логи дублируются в VictoriaMetrics/VictoriaLogs
через OTLP HTTP API (/insert/opentelemetry/v1/logs). Отправка идёт в фоновом
потоке через очередь (drop+warn при переполнении / недоступности).

Формат: Protobuf (ExportLogsServiceRequest). VictoriaLogs v1.x принимает
только protobuf на OTLP endpoint, не JSON — поэтому используется готовый
сгенерированный код из PyPI-пакета opentelemetry-proto.
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

# Готовые OTLP protobuf-сообщения (из pip install opentelemetry-proto).
# Ленивый импорт: загружаем ПРИ ПЕРВОМ создании VictoriaLogsHandler (в __init__),
# НЕ на module-level. Так:
#   - app/main.py импортирует logging_setup БЕЗ protobuf → даже если пакет
#     не установлен или версия несовместима, приложение стартует (просто
#     без remote-логирования). Раньше module-level import убивал старт
#     ImportError'ом → контейнер крашился до первого запроса.
#   - protobuf-зависимость грузится только когда victorialogs_url реально задан.
_pb_logs_service = None
_pb_logs = None
_pb_common = None

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
        self._enabled = self._load_protobuf()
        if not self._enabled:
            # protobuf недоступен — handler бесполезен, не запускаем поток.
            # stderr-логирование продолжает работать через JsonFormatter.
            return
        # urllib с таймаутом, чтобы поток не завис при недоступности.
        self._thread = threading.Thread(
            target=self._run, name="victorialogs-sender", daemon=True
        )
        self._thread.start()

    @staticmethod
    def _load_protobuf() -> bool:
        """Лениво импортирует opentelemetry-proto. Возвращает False, если пакет
        не установлен или версия несовместима — тогда handler self-disable'ится
        (emit становится no-op), stderr-логирование продолжает работать.
        """
        global _pb_logs_service, _pb_logs, _pb_common
        try:
            from opentelemetry.proto.collector.logs.v1 import (
                logs_service_pb2 as svc,
            )
            from opentelemetry.proto.logs.v1 import logs_pb2 as logs
            from opentelemetry.proto.common.v1 import common_pb2 as common
            _pb_logs_service = svc
            _pb_logs = logs
            _pb_common = common
            return True
        except Exception as exc:
            sys.stderr.write(
                f"[victorialogs] opentelemetry-proto unavailable, "
                f"remote logging disabled: {exc}\n"
            )
            sys.stderr.flush()
            return False

    def emit(self, record: logging.LogRecord) -> None:
        if not getattr(self, "_enabled", False):
            return  # protobuf недоступен — no-op, stderr-логирование работает
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
        """Строит Protobuf ExportLogsServiceRequest из батча лог-записей.

        Структура (канонический OTLP): ResourceLogs → ScopeLogs → LogRecord.
        Extra-поля идут как AnyValue attributes (фильтрация/группировка в
        VictoriaLogs через LogsQL).
        """
        now_ns = int(time.time() * 1e9)
        request = _pb_logs_service.ExportLogsServiceRequest()
        resource_logs = request.resource_logs.add()
        resource_logs.resource.attributes.append(_pb_common.KeyValue(
            key="service.name",
            value=_pb_common.AnyValue(string_value=self.service_name),
        ))
        scope_logs = resource_logs.scope_logs.add()
        scope_logs.scope.name = "acuity-yolo"

        for p in batch:
            # Парсим ISO-время записи обратно в ns.
            ts_str = p.get("time", "")
            ts_unix_ns = now_ns
            try:
                # "2026-07-19T..." → unix-секунды.
                dt = time.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S")
                ts_unix_ns = int(time.mktime(dt)) * 10**9
            except (ValueError, OverflowError):
                pass

            level_upper = str(p.get("level", "info")).upper()
            sev_num = _SEVERITY_NUM.get(level_upper, 9)

            log_record = scope_logs.log_records.add()
            log_record.time_unix_nano = ts_unix_ns
            log_record.observed_time_unix_nano = now_ns
            log_record.severity_number = sev_num
            log_record.severity_text = level_upper
            log_record.body.string_value = str(p.get("msg", ""))

            # Extra-поля как attributes (key + AnyValue).
            for key, value in p.items():
                if key in ("msg", "time", "level"):
                    continue
                attr = log_record.attributes.add()
                attr.key = key
                # Все значения стрингифицируем — VictoriaLogs индексирует
                # значения как строки, AnyValue.string_value — самый совместимый
                # путь (числа можно фильтровать через range-синтаксис LogsQL).
                attr.value.string_value = str(value)

        return request.SerializeToString()

    def _post(self, body: bytes) -> bool:
        """POST protobuf-батча в OTLP endpoint. Возвращает True при успехе.

        Покрывает ВСЕ исключения (включая http.client.HTTPException,
        ssl.SSLError и т.п.) и явно закрывает socket при HTTPError.
        Старая версия ловила только (URLError, TimeoutError, OSError) —
        HTTPError хоть и subclass URLError, но несёт свой response object
        (exc.fp), чей socket не закрывался → утечка FD на каждом 4xx/5xx
        (особенно 429 throttling от Cloudflare/proxy) → eventual EMFILE.
        """
        req = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            # VictoriaLogs принимает только protobuf на OTLP endpoint.
            headers={"Content-Type": "application/x-protobuf"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return 200 <= resp.status < 300
        except urllib.error.HTTPError as exc:
            # У HTTPError свой response (exc.fp) — его socket не закрывается
            # with-блоком urlopen. Закрываем явно, иначе каждый 4xx/5xx
            # ответ течёт сокетом (429 throttling → десятки утечек в минуту).
            if exc.fp is not None:
                try:
                    exc.fp.close()
                except Exception:
                    pass
            self._dropped += 1
            self._maybe_warn_dropped()
            sys.stderr.write(f"[victorialogs] POST failed: HTTP {exc.code} {exc.reason}\n")
            sys.stderr.flush()
            return False
        except Exception as exc:
            # Широкий catch: ssl.SSLError, http.client.HTTPException,
            # ConnectionResetError и пр. — любое исключение в лог-пути не
            # должно ронять фоновый поток (см. фикс #3 для пояса безопасности).
            self._dropped += 1
            self._maybe_warn_dropped()
            sys.stderr.write(f"[victorialogs] POST failed: {exc}\n")
            sys.stderr.flush()
            return False

    def _run(self) -> None:
        """Главный цикл фонового потока: копит и флешит батчи.

        Любое исключение в _flush перехватывается — поток НЕ умирает. Раньше
        один сбой (например protobuf-ошибка на несериализуемом значении, или
        исключение вне catch в _post) убивал поток навсегда: очередь
        переполнялась до queue_max и зависала, каждый последующий emit()
        делал put_nowait→Full→stderr-warn → лишний CPU на каждый запрос.
        Теперь разовый сбой батча логируется и цикл продолжается.
        """
        batch: list[dict[str, Any]] = []
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=self.flush_interval)
            except queue.Empty:
                if batch:
                    self._safe_flush(batch)
                    batch = []
                continue
            if item is None:  # sentinel для остановки
                break
            batch.append(item)
            if len(batch) >= self.batch_size:
                self._safe_flush(batch)
                batch = []
        # Финальный flush при остановке.
        if batch:
            self._safe_flush(batch)

    def _safe_flush(self, batch: list[dict[str, Any]]) -> None:
        """Обёртка над _flush: ловит любые исключения, не даёт потоку умереть."""
        try:
            self._flush(batch)
        except Exception as exc:
            # Лог-путь не должен ронять поток. Дропаем проблемный батч целиком
            # и продолжаем — следующий батч отправится нормально.
            sys.stderr.write(f"[victorialogs] flush failed, dropped {len(batch)} records: {exc}\n")
            sys.stderr.flush()

    def _flush(self, batch: list[dict[str, Any]]) -> None:
        try:
            body = self._build_otlp_payload(batch)
        except Exception as exc:
            # Сборка protobuf упала (несериализуемое extra-значение и т.п.) —
            # дропаем батч, не роняя поток. Логируем для диагностики.
            sys.stderr.write(f"[victorialogs] protobuf build failed, dropped {len(batch)} records: {exc}\n")
            sys.stderr.flush()
            return
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
        # Поток запускается только если protobuf загрузился (_enabled=True).
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout=self.flush_interval + 5)
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
    # Чистим хендлеры, ПРЕДВАРИТЕЛЬНО закрывая каждый. Раньше просто
    # removeHandler() бросал ссылку на старый VictoriaLogsHandler — его
    # фоновый поток продолжал жить (orphan) и POST'ить в VictoriaLogs,
    # копились сокеты и дубликаты батчей. Особенно вредно при uvicorn reload
    # или повторных вызовах setup_logging из тестов.
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
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
