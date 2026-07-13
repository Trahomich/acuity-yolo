"""ONNX-инференс YOLO (YOLOv8/v12 и подобные).

Загружает .onnx-модель один раз при старте процесса, держит сессию в памяти.
Препроцессинг — letterbox до INPUT_SIZE×INPUT_SIZE (без искажения пропорций);
постпроцессинг — декодинг выходного тензора ultralytics-формата
[1, 4+nc, num_anchors] + greedy per-class NMS. Координаты боксов
переводятся обратно в пиксели исходного изображения.

Контракт вывода ultralytics YOLOv8/v12 ONNX (default export):
    output[0] shape [1, 4+nc, num_anchors]
    строки 0..3 — cx, cy, w, h (в координатах INPUT_SIZE)
    строки 4..  — class scores (уже после sigmoid; objectness нет в v8+)
Боксы в формате xywh, центр. Анкоры расставлены по сеткам трёх голов
(stride 8/16/32 для 640) — но для вывода это не важно: модель отдаёт готовые
координаты в пикселях input-пространства.

Execution Provider переключается конфигом (cpu по умолчанию). Доступные EP
зависят от установленного пакета: onnxruntime (cpu), onnxruntime-gpu (cuda/
tensorrt), onnxruntime-rocm (rocm). Переключение — без смены модели.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .config import Settings

log = logging.getLogger(__name__)

# COCO-80 в порядке выхода модели (ultralytics default). Используется для
# перевода индекса класса в человекочитаемое имя. При собственной модели с
# другим набором классов — замените на names.json рядом с .onnx (TODO тираж 2).
COCO_NAMES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


@dataclass(slots=True)
class Detection:
    cls_id: int
    cls: str
    score: float
    # xyxy в пикселях исходного изображения.
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


def _providers(preferred: str) -> list[tuple[str, dict[str, Any]] | str]:
    """Возвращает список ONNX Runtime EP в порядке предпочтения.

    Запрошенный EP ставится первым; cpu всегда как фолбэк. Если запрошенный
    EP недоступен в сборке onnxruntime — сессия упадёт с понятной ошибкой при
    старте (лучше fail-fast, чем тихий откат на cpu без уведомления).
    """
    preferred = preferred.lower()
    order: list[tuple[str, dict[str, Any]] | str] = []
    if preferred == "cuda":
        order.append(("CUDAExecutionProvider", {"device_id": 0}))
    elif preferred == "rocm":
        order.append(("ROCMExecutionProvider", {"device_id": 0}))
    elif preferred == "tensorrt":
        order.append(("TensorrtExecutionProvider", {"device_id": 0}))
    elif preferred == "openvino":
        order.append("OpenVINOExecutionProvider")
    order.append("CPUExecutionProvider")
    return order


class YoloModel:
    """Обёртка над ONNX-сессией YOLO с препроцессингом и NMS."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        path = Path(settings.model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found at {path}. "
                "Run `python export_model.py --model yolov12n.pt` to export, "
                "or mount a volume with the .onnx file (see README)."
            )
        providers = _providers(settings.execution_provider)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        log.info("loading onnx model", extra={
            "model_path": str(path), "ep_requested": settings.execution_provider,
        })
        self.session = ort.InferenceSession(
            str(path), sess_options=so, providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        in_shape = self.session.get_inputs()[0].shape  # напр. [1, 3, 640, 640]
        self.input_size = int(in_shape[-1]) if in_shape[-1] not in (None, "None", "?") else settings.input_size

        # Какой EP реально включился.
        active = self.session.get_providers()
        self.active_ep = "cpu"
        for p in active:
            if p != "CPUExecutionProvider":
                self.active_ep = p.replace("ExecutionProvider", "").lower()
                break
        log.info("onnx session ready", extra={
            "input_size": self.input_size, "ep_active": self.active_ep,
            "providers": active,
        })

        # Имена классов: рядом с моделью может лежать names.json (переопределение).
        names_path = path.with_suffix(".names.json")
        if names_path.is_file():
            import json
            with names_path.open("r", encoding="utf-8") as f:
                self.names = json.load(f)
        else:
            self.names = COCO_NAMES

    # ------------------------------------------------------------------
    # Препроцессинг
    # ------------------------------------------------------------------

    def letterbox(self, img: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """Ресайз без искажения пропорций + pad до input_size×input_size.

        Возвращает (padded[1,3,H,W] float32 RGB /255, scale, pad_w, pad_h).
        """
        h, w = img.shape[:2]
        size = self.input_size
        scale = min(size / h, size / w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = (size - new_w) / 2.0
        pad_h = (size - new_h) / 2.0
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        # HWC BGR → CHW RGB float32 /255. Вход модели — RGB (ultralytics export).
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        padded = np.ascontiguousarray(padded[None, ...])
        return padded, scale, pad_w, pad_h

    # ------------------------------------------------------------------
    # Постпроцессинг
    # ------------------------------------------------------------------

    def postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad_w: float,
        pad_h: float,
        orig_h: int,
        orig_w: int,
        conf: float,
        iou: float,
    ) -> list[Detection]:
        """Декодинг + NMS вывода ultralytics YOLOv8/v12 ONNX.

        output: [1, 4+nc, num_anchors] или [1, num_anchors, 4+nc].
        """
        preds = output[0]
        if preds.shape[0] != 4 and preds.shape[-1] in (4 + len(self.names),):
            # формат [num_anchors, 4+nc] — транспонируем.
            preds = preds.T
        # preds: [4+nc, num_anchors]
        nc = len(self.names)
        if preds.shape[0] != 4 + nc:
            # Несовпадение числа классов: обрезаем/дополняем по минимуму.
            nc = preds.shape[0] - 4
            if nc <= 0:
                return []

        boxes_xywh = preds[:4, :]           # cx, cy, w, h (input-пространство)
        scores_all = preds[4:4 + nc, :]     # [nc, num_anchors]

        # Лучший класс на каждый анкор.
        cls_ids = scores_all.argmax(axis=0)            # [num_anchors]
        max_scores = scores_all.max(axis=0)            # [num_anchors]

        mask = max_scores >= conf
        if not np.any(mask):
            return []

        cls_ids = cls_ids[mask]
        max_scores = max_scores[mask]
        boxes_xywh = boxes_xywh[:, mask]               # [4, K]

        # xywh → xyxy в input-пространстве, затем обратный letterbox.
        cx, cy, w, h = boxes_xywh[0], boxes_xywh[1], boxes_xywh[2], boxes_xywh[3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        # Снимаем letterbox: вычитаем pad, делим на scale, клипаем в кадр.
        # Координаты — векторы длины K (по числу боксов); clip векторный.
        x1 = np.clip((x1 - pad_w) / scale, 0, orig_w)
        y1 = np.clip((y1 - pad_h) / scale, 0, orig_h)
        x2 = np.clip((x2 - pad_w) / scale, 0, orig_w)
        y2 = np.clip((y2 - pad_h) / scale, 0, orig_h)

        # NMS per-class.
        keep = self._nms_per_class(
            x1, y1, x2, y2, cls_ids.astype(np.int64), max_scores, iou,
        )
        out: list[Detection] = []
        for idx in keep:
            cid = int(cls_ids[idx])
            name = self.names[cid] if 0 <= cid < len(self.names) else f"cls{cid}"
            out.append(Detection(
                cls_id=cid, cls=name, score=float(max_scores[idx]),
                x1=float(x1[idx]), y1=float(y1[idx]),
                x2=float(x2[idx]), y2=float(y2[idx]),
            ))
        return out

    @staticmethod
    def _nms_per_class(
        x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray,
        cls_ids: np.ndarray, scores: np.ndarray, iou_thr: float,
    ) -> list[int]:
        """Greedy NMS отдельно по каждому классу. Возвращает индексы keep."""
        keep: list[int] = []
        for c in np.unique(cls_ids):
            sel = np.where(cls_ids == c)[0]
            order = sel[np.argsort(-scores[sel])]
            xx1 = x1[order]
            yy1 = y1[order]
            xx2 = x2[order]
            yy2 = y2[order]
            area = (xx2 - xx1) * (yy2 - yy1)
            suppressed = np.zeros(len(order), dtype=bool)
            for i in range(len(order)):
                if suppressed[i]:
                    continue
                keep.append(int(order[i]))
                ix1 = np.maximum(xx1[i], xx1[i + 1:])
                iy1 = np.maximum(yy1[i], yy1[i + 1:])
                ix2 = np.minimum(xx2[i], xx2[i + 1:])
                iy2 = np.minimum(yy2[i], yy2[i + 1:])
                iw = np.maximum(0.0, ix2 - ix1)
                ih = np.maximum(0.0, iy2 - iy1)
                inter = iw * ih
                union = area[i] + area[i + 1:] - inter
                iou = np.where(union > 0, inter / union, 0.0)
                suppressed[i + 1:] |= iou > iou_thr
        return keep

    # ------------------------------------------------------------------
    # Полный пайплайн
    # ------------------------------------------------------------------

    def detect(self, image_bgr: np.ndarray, conf: float | None = None,
               iou: float | None = None) -> tuple[list[Detection], int]:
        """Инференс одного кадра. Возвращает (детекции, inference_ms)."""
        conf = self.settings.conf_threshold if conf is None else conf
        iou = self.settings.iou_threshold if iou is None else iou
        orig_h, orig_w = image_bgr.shape[:2]

        inp, scale, pad_w, pad_h = self.letterbox(image_bgr)
        t0 = time.perf_counter()
        output = self.session.run(None, {self.input_name: inp})[0]
        inference_ms = int((time.perf_counter() - t0) * 1000)

        dets = self.postprocess(
            output, scale, pad_w, pad_h, orig_h, orig_w, conf, iou,
        )
        return dets, inference_ms
