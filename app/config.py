"""Конфигурация inference-воркера acuity-yolo.

Все параметры читаются из переменных окружения (12-factor). Источник по
умолчанию — configs/worker.yml; в Docker значения прокидываются через env
(см. docker-compose.yml). Приоритет: env > worker.yml > дефолт кода.

Ключевые параметры:
  MODEL_PATH           — путь к .onnx файлу модели (монтируется volume).
  EXECUTION_PROVIDER   — cpu (по умолчанию) | cuda | rocm | openvino.
  CONF_THRESHOLD       — порог уверенности бокса (0..1).
  IOU_THRESHOLD        — порог IoU для NMS (0..1).
  INPUT_SIZE           — размер входа модели (квадрат, напр. 640).
  WORKERS              — число uvicorn-воркеров (процессов) в контейнере.
                         Для горизонтального масштабирования под нагрузкой
                         поднимайте число реплик контейнера (scale), а не
                         только воркеров — модель грузится в каждый процесс.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_yaml_defaults() -> dict:
    """Читает configs/worker.yml как источник значений по умолчанию.

    Файл опционален: если его нет — используются дефолты кода, перекрываемые
    переменными окружения. Имя файла берётся из ACUITY_WORKER_CONFIG
    (по умолчанию configs/worker.yml относительно рабочей директории).
    """
    path = Path(os.getenv("ACUITY_WORKER_CONFIG", "configs/worker.yml"))
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # flatten: секция worker → верхний уровень.
    return {k.upper(): v for k, v in (data.get("worker", data) or {}).items()}


ExecutionProvider = Literal["cpu", "cuda", "rocm", "openvino", "tensorrt"]


class Settings(BaseSettings):
    """Настройки воркера. Pydantic-settings: env перекрывает yaml-дефолты."""

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    model_path: str = Field(default="models/yolov12m.onnx")
    execution_provider: ExecutionProvider = Field(default="cpu")
    conf_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    input_size: int = Field(default=640, ge=32, le=4096)

    # Имя модели для ответа detections.model (метка для логики/метрик).
    model_name: str = Field(default="yolov12m")

    # HTTP-сервер.
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1, ge=1)

    # Уровень логирования.
    log_level: str = Field(default="info")

    # Максимальный размер загружаемого изображения (байт); защита от OOM.
    max_image_bytes: int = Field(default=20 * 1024 * 1024)

    @property
    def model_file(self) -> Path:
        return Path(self.model_path)

    @property
    def model_ready(self) -> bool:
        return self.model_file.is_file()


def _build_settings() -> Settings:
    """Собирает Settings: env-переменные имеют приоритет над yaml-дефолтами.

    Pydantic читает env напрямую; yaml подкладывается как дефолты через
    init-kwargs, чтобы env побеждал.
    """
    defaults = _load_yaml_defaults()
    # Только строковые/числовые скаляры из yaml как дефолты; env в BaseSettings
    # имеет приоритет над init-значениями при env_prefix="".
    return Settings(**defaults)  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Синглтон-настройки (кешируются на время жизни процесса)."""
    return _build_settings()
