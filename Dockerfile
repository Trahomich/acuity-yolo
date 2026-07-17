# syntax=docker/dockerfile:1.6
# Dockerfile — образ inference-воркера acuity-yolo.
#
# Multi-stage:
#   1. builder-{cpu|gpu-amd} — venv с onnxruntime (cpu) или onnxruntime-rocm (gpu-amd)
#   2. exporter              — (опционально) экспорт YOLOv12m.pt → ONNX
#   3. runtime-{cpu|gpu-amd} — финальный образ; выбирается через FROM runtime-${DEVICE}
#
# DEVICE=cpu     → python:3.11-slim (лёгкий, переносимый). onnxruntime с PyPI.
# DEVICE=gpu-amd → rocm/dev-ubuntu-22.04:6.4-complete (полный ROCm 6.4, все
#                  HIP-библиотеки и symlink-ферма внутри — libhipblas.so.2 и т.п.
#                  на месте). onnxruntime-rocm 1.21.0 (cp310) с repo.radeon.com
#                  под системный Python 3.10 Ubuntu 22.04.
#                  Образ самодостаточен, не зависит от того, что стоит на хосте.
#
# Задел: DEVICE=gpu-nvidia — аналогично через nvidia/cuda-образы + onnxruntime-gpu.
#
# Шаг exporter управляется EXPORT_MODEL (по умолчанию 1).
# EP в рантайме задаётся EXECUTION_PROVIDER и должно совпадать с DEVICE.

# Глобальный ARG — виден во всех stages, включая FROM runtime-${DEVICE}.
ARG DEVICE=cpu

############################
# Builder (CPU): python:3.11-slim
############################
FROM python:3.11-slim AS builder-cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

############################
# Builder (AMD GPU): rocm/dev-ubuntu-22.04:6.4-complete, системный Python 3.10
############################
FROM rocm/dev-ubuntu-22.04:6.4-complete AS builder-gpu-amd

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python 3.10 уже в образе (Ubuntu 22.04), но модуль venv вынесен в отдельный
# пакет python3.10-venv — без него `python3 -m venv` падает.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3.10-venv python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip \
    && python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY requirements-gpu-amd.txt scripts/clear_execstack.py ./
RUN pip install --upgrade pip && pip install -r requirements-gpu-amd.txt && \
    # onnxruntime .so имеет PT_GNU_STACK=RWE (executable stack). Docker seccomp
    # режет mprotect(PROT_EXEC) → ImportError. Снимаем X-бит через Python
    # (execstack/prelink убраны из Debian). https://github.com/microsoft/onnxruntime/issues/24911
    python clear_execstack.py

############################
# Exporter: YOLOv12m.pt → ONNX (опционально)
############################
FROM python:3.11-slim AS exporter

ARG EXPORT_MODEL=1
ARG MODEL_SPEC=yolo12m.pt
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
RUN mkdir -p /export/models

# Если EXPORT_MODEL=1 — ставим ultralytics (CPU-only torch) и экспортируем YOLO.
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
# Runtime — CPU
############################
FROM python:3.11-slim AS runtime-cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ACUITY_WORKER_CONFIG=/etc/acuity-yolo/configs/worker.yml

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 yolo \
    && useradd --system --uid 10001 --gid yolo --no-create-home --home-dir /app yolo

COPY --from=builder-cpu /opt/venv /opt/venv

WORKDIR /app
COPY app/ ./app/
COPY configs/ /etc/acuity-yolo/configs/
COPY export_model.py ./
COPY --from=exporter --chown=yolo:yolo /export/models/ ./models/

USER yolo
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

############################
# Runtime — AMD GPU (ROCm)
############################
FROM rocm/dev-ubuntu-22.04:6.4-complete AS runtime-gpu-amd

# Базовый образ содержит /opt/rocm-6.4.0 с полной HIP-библиотечной фермой ROCm 6.4
# (libamdhip64, libhipblas.so.2, libMIOpen и т.д.) + symlink /opt/rocm → версия.
# onnxruntime-rocm 1.21.0 (cp310) собран под эту версию — soname совпадают.
# Дополнительных apt-установок HIP-библиотек НЕ требуется — всё в образе.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ACUITY_WORKER_CONFIG=/etc/acuity-yolo/configs/worker.yml \
    LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/llvm/lib" \
    # RDNA3 (gfx1100) иногда определяется как unsupported — фиксируем явно.
    HSA_OVERRIDE_GFX_VERSION="11.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 yolo \
    && useradd --system --uid 10001 --gid yolo --no-create-home --home-dir /app yolo

# venv из builder-gpu-amd (тот же базовый образ + системный Python 3.10).
COPY --from=builder-gpu-amd /opt/venv /opt/venv

WORKDIR /app
COPY app/ ./app/
COPY configs/ /etc/acuity-yolo/configs/
COPY export_model.py ./
COPY --from=exporter --chown=yolo:yolo /export/models/ ./models/

USER yolo
EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

############################
# Final: выбираем runtime по DEVICE
############################
ARG DEVICE
FROM runtime-${DEVICE} AS runtime
