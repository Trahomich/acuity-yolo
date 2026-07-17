"""Pydantic-схемы ответов воркера.

Контракт /detect соответствует ТЗ docs/EVENT_CLASSIFICATION.md §5.2:

    POST /detect (multipart: image=JPEG)
    → 200 {
        "boxes": [{"cls","cls_id","score","x","y","w","h"}],
        "classes": ["person"],
        "inference_ms": 47,
        "ep": "cpu"
      }

Координаты боксов — в пикселях исходного изображения (до letterbox-обратного
преобразования): (x, y) — верхний-левый угол, (w, h) — ширина/высота.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Box(BaseModel):
    cls: str = Field(description="Имя класса COCO, напр. 'person'")
    cls_id: int = Field(description="Числовой ID класса COCO (0..79)")
    score: float = Field(description="Уверенность 0..1")
    x: int = Field(description="X верхнего-левого угла (px исходного кадра)")
    y: int = Field(description="Y верхнего-левого угла (px исходного кадра)")
    w: int = Field(description="Ширина бокса (px)")
    h: int = Field(description="Высота бокса (px)")


class DetectResponse(BaseModel):
    boxes: list[Box] = Field(default_factory=list)
    classes: list[str] = Field(
        default_factory=list,
        description="Уникальные имена классов на кадре (для быстрого тегирования)",
    )
    inference_ms: int = Field(description="Время инференса (мс), без препроцессинга I/O")
    model: str = Field(description="Метка модели (из конфига)")
    ep: str = Field(description="Execution Provider: cpu/cuda/rocm/...)")


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    model_ready: bool
    ep: str


class ErrorResponse(BaseModel):
    detail: str
