# Dockerfile — образ inference-воркера acuity-yolo.
#
# Multi-stage:
#   1. builder     — ставит runtime-зависимости в venv
#   2. exporter    — (опционально) экспортирует YOLOv12n.pt → ONNX через ultralytics
#   3. runtime     — python:3.11-slim + venv + модель + app/
#
# Шаг exporter управляется build-arg EXPORT_MODEL (по умолчанию 1). Чтобы
# отключить экспорт (модель монтируется volume или копируется вручную), собирайте:
#   docker build --build-arg EXPORT_MODEL=0 -t acuity-yolo .
# Если экспорт включён, но интернета нет — сборка упадёт на pip install
# ultralytics; используйте EXPORT_MODEL=0 и примонтируйте .onnx через volume.
#
# Execution Provider:
#   CPU (по умолчанию):  onnxruntime          — собирается как есть.
#   GPU (CUDA/ROCm):     собрать с другим requirements-gpu.txt или поставить
#                         onnxruntime-gpu/-rocm, и задать EXECUTION_PROVIDER.
#   В тираже 1 (ТЗ) стартуем на CPU; GPU-EP — конфигом, без пересборки логики.

############################
# Builder: venv с зависимостями
############################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

############################
# Exporter: YOLOv12n.pt → ONNX (опциональный шаг)
############################
FROM python:3.11-slim AS exporter

ARG EXPORT_MODEL=1
ARG MODEL_SPEC=yolov12n.pt
ARG IMGSZ=640
ARG OPSET=12

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /export

# Если EXPORT_MODEL=1 — ставим ultralytics и экспортируем. Иначе оставляем
# пустой models/ (воркер стартует not-ready, модель примонтируется volume).
RUN if [ "$EXPORT_MODEL" = "1" ]; then \
        pip install --upgrade pip && pip install ultralytics && \
        python -c "from ultralytics import YOLO; \
                   m=YOLO('${MODEL_SPEC}'); \
                   m.export(format='onnx', imgsz=${IMGSZ}, opset=${OPSET}, simplify=True, dynamic=False, half=False)" && \
        mkdir -p /export/models && \
        cp /root/*.onnx /export/models/ 2>/dev/null || cp *.onnx /export/models/ 2>/dev/null || \
        find / -name "*.onnx" -not -path "/proc/*" -exec cp {} /export/models/ \; ; \
    else \
        mkdir -p /export/models && touch /export/models/.skip ; \
    fi

############################
# Runtime
############################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ACUITY_WORKER_CONFIG=/etc/acuity-yolo/configs/worker.yml

# Минимальный runtime: libGL для opencv-headless (libglib — зависимость).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 yolo \
    && useradd --system --uid 10001 --gid yolo --no-create-home --home-dir /app yolo

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app/ ./app/
COPY configs/ /etc/acuity-yolo/configs/
COPY export_model.py ./

# Модель из exporter-стадии (может быть пустой, если EXPORT_MODEL=0).
COPY --from=exporter --chown=yolo:yolo /export/models/ ./models/

USER yolo
EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
# workers=1 по умолчанию; для нагрузки поднимайте число реплик (--scale),
# а не uvicorn-воркеров — модель в каждом процессе.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
