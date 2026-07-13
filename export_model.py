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
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO to ONNX for acuity-yolo.")
    parser.add_argument("--model", default="yolov12n.pt",
                        help="ultralytics model spec (default: yolov12n.pt)")
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

    print(f"[export] loading {args.model} (imgsz={args.imgsz}, opset={args.opset})")
    model = YOLO(args.model)
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
    src = Path(path)
    if src.is_file() and src.resolve().parent != out_dir.resolve():
        dst = out_dir / src.name
        src.replace(dst)
        print(f"[export] moved → {dst}")
        path = dst
    print(f"[export] done: {path}")
    print(f"[export] для запуска воркера: MODEL_PATH={path} python -m uvicorn app.main:app")


if __name__ == "__main__":
    main()
