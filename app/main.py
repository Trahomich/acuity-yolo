"""FastAPI inference-воркер acuity-yolo.

Stateless REST-сервис: модель грузится в память процесса один раз при старте;
каждый POST /detect самодостаточен (multipart JPEG). Никаких шин (Kafka/NATS),
никакого общего состояния между запросами — горизонтальное масштабирование
через N реплик за балансером (nginx/HAProxy round-robin, Docker --scale).

Эндпоинты:
    POST /detect   — multipart image=JPEG → DetectResponse (ТЗ §5.2)
    GET  /healthz  — liveness (200 всегда, если процесс жив)
    GET  /readyz   — readiness (200 если модель загружена, иначе 503)
    GET  /         — краткая сводка состояния

Изоляция: воркер НЕ ходит в NATS/Postgres/S3. Только принимает JPEG и возвращает
боксы. Оркестрацию (забор кадров из S3, запись детекций в БД, тегирование)
делает inference-service (Go) — он же HTTP-клиент к этому воркеру.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import anyio
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .logging_setup import setup_logging
from .model import YoloModel
from .schemas import Box, DetectResponse, ErrorResponse, HealthResponse

log = logging.getLogger("acuity-yolo")

# Глобальное состояние процесса: модель + настройки. Живёт всё время работы.
_state: dict[str, object] = {"model": None}

# CapacityLimiter для инференса: строго ОДНА session.run() concurrently.
# Исключает MIOpen-гонку (find vs forward) и при этом освобождает event loop —
# /healthz и приём новых запросов не подвисают на инференсе.
# Параллелизм по-прежнему достигается uvicorn-воркерами/репликами.
_INFER_LIMITER = anyio.CapacityLimiter(1)


def _get_model() -> YoloModel:
    model = _state.get("model")
    if model is None:  # pragma: no cover — защита от программной ошибки
        raise RuntimeError("model not initialized")
    return model  # type: ignore[return-value]


def _load_model(settings: Settings) -> YoloModel | None:
    """Пытается загрузить модель. Возвращает None (без падения), если файла нет:
    тогда /detect возвращает 503, а /healthz остаётся 200 (процесс жив, но
    not-ready). Так воркер стартует раньше, чем смонтировали модель.
    """
    try:
        return YoloModel(settings)
    except FileNotFoundError as exc:
        log.warning("model not loaded, /detect unavailable", extra={"reason": str(exc)})
        return None
    except Exception:  # pragma: no cover — битая модель
        log.exception("model load failed")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(
        level=settings.log_level,
        victorialogs_url=settings.victorialogs_url,
        service_name=settings.log_service_name,
        flush_interval=settings.log_flush_interval,
        batch_size=settings.log_batch_size,
        queue_max=settings.log_queue_max,
    )
    log.info("acuity-yolo starting", extra={
        "model_path": settings.model_path,
        "ep_requested": settings.execution_provider,
        "workers": settings.workers,
        "pid": os.getpid(),
    })
    _state["model"] = _load_model(settings)
    log.info("acuity-yolo ready", extra={
        "model_ready": _state["model"] is not None,
    })
    yield
    log.info("acuity-yolo shutting down")


app = FastAPI(
    title="acuity-yolo",
    description="Stateless YOLO (YOLOv12) ONNX inference worker for acuity event classification.",
    version="0.1.0",
    lifespan=lifespan,
)


def _decode_image(raw: bytes, max_bytes: int, max_pixels: int) -> np.ndarray:
    if not raw:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="empty image")
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"image too large: {len(raw)} bytes (max {max_bytes})",
        )
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="cannot decode image (expected JPEG/PNG)",
        )
    # Защита от decompression-bomb: компактный файл может распаковаться в
    # гигантский массив пикселей → OOM. Ограничиваем H*W.
    h, w = img.shape[:2]
    if h * w > max_pixels:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"decoded image too large: {w}x{h} = {h*w} pixels (max {max_pixels})",
        )
    return img


@app.post(
    "/detect",
    response_model=DetectResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Плохое изображение"},
        413: {"model": ErrorResponse, "description": "Слишком большой файл"},
        503: {"model": ErrorResponse, "description": "Модель не загружена"},
    },
    summary="Детекция объектов на одном кадре",
)
async def detect(image: UploadFile = File(...)) -> JSONResponse:
    settings = get_settings()
    model = _state.get("model")
    if model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": f"model not loaded; expected at {settings.model_path}"},
        )

    # Читаем тело запроса и сразу закрываем UploadFile — освобождаем spool
    # (tempfile в /tmp при upload >1 MiB). Раньше не закрывался: под нагрузкой
    # и при client-disconnect (Cloudflare 524) temp-файлы копились → FD leak.
    try:
        raw = await image.read()
    finally:
        await image.close()

    # decode + inference — blocking (cv2 + onnxruntime). Выполняем в потоке,
    # но строго ОДНОВРЕМЕННО активных (CapacityLimiter(1)). Это решает обе
    # проблемы сразу:
    #   1) event loop свободен — /healthz, /readyz, приём новых запросов не
    #      подвисают на инференсе (раньше синхронный session.run блокировал
    #      всё → nginx/Cloudflare healthcheck таймаутил → upstream «умирал»).
    #   2) MIOpen-гонка исключена: лимитер гарантирует максимум одну
    #      session.run() concurrently, поэтому find-фаза и forward-фаза
    #      не пересекаются (в прошлый раз asyncio.to_thread без лимита порождал
    #      "No invoker registered for conv", MIOPEN failure 7).
    def _decode_and_detect() -> tuple[list, int]:
        img = _decode_image(raw, settings.max_image_bytes, settings.max_image_pixels)
        return model.detect(img)  # type: ignore[union-attr]

    try:
        dets, inference_ms = await anyio.to_thread.run_sync(
            _decode_and_detect, limiter=_INFER_LIMITER
        )
    except HTTPException:
        raise  # 400/413 из _decode_image — пробрасываем как есть
    except Exception:
        log.exception("inference failed")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "inference error"},
        )

    boxes: list[Box] = []
    classes_seen: set[str] = set()
    for d in dets:
        boxes.append(Box(
            cls=d.cls,
            cls_id=d.cls_id,
            score=round(d.score, 4),
            x=int(round(d.x1)),
            y=int(round(d.y1)),
            w=int(round(d.w)),
            h=int(round(d.h)),
        ))
        classes_seen.add(d.cls)

    # INFO: только агрегаты (быстро, без сериализации боксов на каждый запрос).
    # DEBUG: полный список детекций для отладки (включается LOG_LEVEL=debug).
    log_extra: dict = {
        "inference_ms": inference_ms,
        "boxes_count": len(boxes),
        "classes": sorted(classes_seen) if classes_seen else [],
    }
    if log.isEnabledFor(logging.DEBUG):
        log_extra["detections"] = [
            {
                "cls": b.cls,
                "cls_id": b.cls_id,
                "score": b.score,
                "x": b.x, "y": b.y, "w": b.w, "h": b.h,
            }
            for b in boxes
        ]
    log.info("detect", extra=log_extra)

    resp = DetectResponse(
        boxes=boxes,
        classes=sorted(classes_seen),
        inference_ms=inference_ms,
        model=settings.model_name,
        ep=settings.execution_provider,
    )
    return JSONResponse(status_code=status.HTTP_200_OK, content=resp.model_dump())


@app.get("/healthz", response_model=HealthResponse, summary="Liveness")
async def healthz() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        model=settings.model_name,
        model_ready=_state.get("model") is not None,
        ep=settings.execution_provider,
    )


@app.get("/readyz", response_model=HealthResponse, summary="Readiness")
async def readyz() -> JSONResponse:
    settings = get_settings()
    ready = _state.get("model") is not None
    body = HealthResponse(
        status="ready" if ready else "not-ready",
        model=settings.model_name,
        model_ready=ready,
        ep=settings.execution_provider,
    )
    code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=code, content=body.model_dump())


@app.get("/", response_model=HealthResponse, summary="Корень")
async def root() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        model=settings.model_name,
        model_ready=_state.get("model") is not None,
        ep=settings.execution_provider,
    )
