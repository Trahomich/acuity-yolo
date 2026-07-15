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
#   DEVICE=cpu (по умолчанию для исходного образа): requirements.txt → onnxruntime.
#   DEVICE=rocm (AMD GPU): requirements-rocm.txt → onnxruntime-rocm + HIP-библиотеки.
#   Готовый образ зависит от DEVICE; переключение требует пересборки. EP в рантайме
#   задаётся через EXECUTION_PROVIDER и должно совпадать с DEVICE.

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

# DEVICE=cpu (по умолчанию) → requirements.txt (onnxruntime).
# DEVICE=rocm (AMD GPU)      → requirements-rocm.txt (onnxruntime-rocm + numpy<2).
# HIP-библиотеки (libamdhip64.so, librocm-core.so) НЕ ставим из apt: этих
# пакетов нет в Debian-репозиториях. Их берём с хоста через volume
# /opt/rocm:/opt/rocm:ro в docker-compose.yml, а путь к ним подсказываем
# рантайму через LD_LIBRARY_PATH (см. runtime-стадию ниже).
ARG DEVICE=cpu
COPY requirements.txt requirements-rocm.txt ./
RUN REQ=$([ "$DEVICE" = "rocm" ] && echo requirements-rocm.txt || echo requirements.txt) && \
    pip install --upgrade pip && pip install -r "$REQ"

############################
# Exporter: YOLOv12n.pt → ONNX (опциональный шаг)
############################
FROM python:3.11-slim AS exporter

ARG EXPORT_MODEL=1
ARG MODEL_SPEC=yolo12n.pt
ARG IMGSZ=640
ARG OPSET=12

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /export
COPY export_model.py ./

# Системные библиотеки для opencv-python-headless (ultralytics тянет cv2):
# без libxcb/libGL импорт cv2 падает на этапе экспорта.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Готовим целевой каталог заранее (COPY --from=exporter требует его наличия
# даже если экспорт выключен — тогда воркер стартует not-ready).
RUN mkdir -p /export/models

# Если EXPORT_MODEL=1 — ставим ultralytics (CPU-only torch, без CUDA-депов)
# и экспортируем YOLO в ONNX через export_model.py (он падает с ненулевым
# exit при ошибке импорта/экспорта, копирует .onnx в --out). Иначе каталог
# остаётся пустым — воркер стартует not-ready, модель примонтируется volume
# или кладётся вручную.
#
# torch CPU-only ставим с PyPI-индекса CPU-сборок, чтобы не тянуть ~3 ГБ
# CUDA-пакетов (нам для экспорта/инференса нужен только CPU). Порядок важен:
# сначала torch CPU, потом ultralytics (он не переустановит torch, т.к. тот
# уже удовлетворяет зависимости).
RUN if [ "$EXPORT_MODEL" = "1" ]; then \
        pip install --upgrade pip && \
        pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
        pip install ultralytics onnx onnxslim && \
        python export_model.py --model "${MODEL_SPEC}" --imgsz "${IMGSZ}" --opset "${OPSET}" --out /export/models && \
        ls -la /export/models/; \
    else \
        echo "EXPORT_MODEL=0, skipping export (models/ stays empty; worker starts not-ready)"; \
    fi

############################
# Runtime
############################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ACUITY_WORKER_CONFIG=/etc/acuity-yolo/configs/worker.yml \
    # onnxruntime-rocm ищет libamdhip64.so / librochook.so в /opt/rocm/lib —
    # они приходят с хоста через volume (см. docker-compose.yml). Без этого
    # падает "libamdhip64.so: cannot open shared object file" при старте EP.
    LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/llvm/lib:${LD_LIBRARY_PATH}" \
    # HSA_OVERRIDE_GFX_version: ROCm на RDNA3 (gfx1100) иногда определяется
    # как неsupported; фиксируем явно. Для других GPU — переопределить.
    HSA_OVERRIDE_GFX_VERSION="11.0.0"

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
