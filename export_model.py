#!/usr/bin/env python3
"""Экспорт YOLO-модели (YOLOv8/v12) в ONNX для acuity-yolo воркера.

Запуск (нужен интернет для первой загрузки весей; кешируется в ~/.cache):

    pip install ultralytics
    python export_model.py                          # → models/yolov12n.onnx
    python export_model.py --model yolov8n.pt       # явная модель
    python export_model.py --imgsz 640 --opset 12

Модель по умолчанию — yolov12n (ТЗ §10, зафиксированное решение). ultralytics
скачает веса автоматически при первом запуске. Размер входа — 640 (ТЗ §5.2).
Экспорт — dynamic=False (фиксированный 640×640), simplify=True.

Этот скрипт НЕ входит в requirements.txt воркера (ultralytics тяжёлый и нужен
только для экспорта). Запускайте локально или в отдельном build-step.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

# Базовый URL весей YOLO на GitHub releases ultralytics. Имя файла = имя модели
# в нотации ultralytics: yolo12n.pt (YOLOv12), yolov8n.pt (v8) и т.д. Веса для
# всех поколений лежат в теге v8.3.0 единообразно. ВАЖНО: YOLO12 называется
# yolo12n (без 'v'), не yolov12n — см. https://docs.ultralytics.com/models/yolo12.
_WEIGHTS_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0/"


def _ensure_weights(model_spec: str) -> Path:
    """Возвращает путь к .pt-файлу; при необходимости скачивает его.

    model_spec может быть именем ('yolov12n.pt') или путём к существующему
    файлу. Голое имя без расширения дополняется .pt.
    """
    p = Path(model_spec)
    # Дополняем .pt, если задано голое имя.
    if p.suffix == "" and p.stem != "":
        p = p.with_suffix(".pt")
    if p.is_file():
        print(f"[export] weights found locally: {p}")
        return p
    if p.name != str(p):
        # Задан путь, но файла нет — это ошибка (не имя модели).
        raise SystemExit(f"[export] weights file not found: {p}")
    # Голое имя модели — скачиваем.
    url = _WEIGHTS_BASE + p.name
    print(f"[export] downloading weights {p.name} from {url}")
    try:
        urllib.request.urlretrieve(url, p.name)
    except Exception as exc:
        raise SystemExit(f"[export] cannot download {p.name} from {url}: {exc}")
    print(f"[export] downloaded {p.name} ({Path(p.name).stat().st_size} bytes)")
    return Path(p.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO to ONNX for acuity-yolo.")
    parser.add_argument("--model", default="yolo12n.pt",
                        help="ultralytics model spec (default: yolo12n.pt — YOLOv12)")
    parser.add_argument("--imgsz", type=int, default=640, help="input size (default 640)")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset (default 12)")
    parser.add_argument("--out", default="models",
                        help="output directory (default: models)")
    parser.add_argument("--half", action="store_true",
                        help="FP16 export (только для GPU; см. README)")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics не установлен. Установите: pip install ultralytics"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Если веса .pt не лежат локально — скачиваем с GitHub releases ultralytics.
    # Ultralytics умеет сам тянуть свои модели, но в чистом окружении без кеша
    # (Docker build) это не всегда срабатывает; скачиваем явно.
    model_path = _ensure_weights(args.model)

    print(f"[export] loading {model_path} (imgsz={args.imgsz}, opset={args.opset})")
    model = YOLO(str(model_path))
    path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
        half=args.half,
    )
    print(f"[export] exported → {path}")
    # ultralytics кладёт <name>.onnx рядом с <name>.pt; переносим в out_dir.
    # Каноническое имя выхода — yolov12n.onnx (под MODEL_PATH воркера по
    # умолчанию), независимо от того, из какой спеки весей экспортировали.
    src = Path(path)
    dst = out_dir / "yolov12n.onnx"
    if src.is_file() and src.resolve() != dst.resolve():
        src.replace(dst)
        print(f"[export] moved → {dst}")
    path = dst
    print(f"[export] done: {path}")
    print(f"[export] для запуска воркера: MODEL_PATH={path} python -m uvicorn app.main:app")


if __name__ == "__main__":
    main()
