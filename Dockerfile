# Dockerfile — образ inference-воркера acuity-yolo.
#
# Multi-stage:
#   1. builder     — ставит runtime-зависимости в venv (python:3.12-slim)
#   2. exporter    — (опционально) экспортирует YOLOv12m.pt → ONNX через ultralytics
#   3. runtime     — финальный образ; зависит от DEVICE:
#                      cpu  → python:3.12-slim (лёгкий, переносимый)
#                      rocm → rocm/dev-ubuntu-22.04:6.4 (полный ROCm 6.4,
#                             все HIP-библиотеки нужных версий внутри, ~1.5 ГБ)
#
# Шаг exporter управляется build-arg EXPORT_MODEL (по умолчанию 1). Чтобы
# отключить экспорт (модель монтируется volume или копируется вручную), собирайте:
#   docker build --build-arg EXPORT_MODEL=0 -t acuity-yolo .
#
# Execution Provider:
#   DEVICE=cpu → onnxruntime (CPU), python:3.12-slim.
#   DEVICE=rocm → onnxruntime-rocm 1.21.0 (ROCm 6.4), базовый образ rocm/dev.
#   ROCm-вариант самодостаточен: HIP-библиотеки внутри образа, НЕ зависит от
#   того, что установлено на хосте. Нужен только amdgpu-драйвер + /dev/kfd,/dev/dri.
#   EP в рантайме задаётся EXECUTION_PROVIDER и должно совпадать с DEVICE.

############################
# Builder: venv с зависимостями (общий для cpu/rocm)
############################
FROM python:3.12-slim AS builder

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

# DEVICE=cpu → requirements.txt (onnxruntime).
# DEVICE=rocm → requirements-rocm.txt (onnxruntime-rocm + numpy<2).
ARG DEVICE=cpu
COPY requirements.txt requirements-rocm.txt scripts/clear_execstack.py ./
RUN REQ=$([ "$DEVICE" = "rocm" ] && echo requirements-rocm.txt || echo requirements.txt) && \
    pip install --upgrade pip && pip install -r "$REQ" && \
    # ROCm-сборка onnxruntime имеет .so с PT_GNU_STACK=RWE (executable stack).
    # Docker seccomp режет mprotect(PROT_EXEC) → ImportError при загрузке.
    # Снимаем X-бит напрямую через Python (execstack/prelink убраны из Debian).
    # Только для DEVICE=rocm. https://github.com/microsoft/onnxruntime/issues/24911
    if [ "$DEVICE" = "rocm" ]; then python clear_execstack.py; fi

############################
# Exporter: YOLOv12m.pt → ONNX (опциональный шаг)
############################
FROM python:3.12-slim AS exporter

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

# Готовим целевой каталог заранее (COPY --from=exporter требует его наличия
# даже если экспорт выключен — тогда воркер стартует not-ready).
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
FROM python:3.12-slim AS runtime-cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ACUITY_WORKER_CONFIG=/etc/acuity-yolo/configs/worker.yml

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 yolo \
    && useradd --system --uid 10001 --gid yolo --no-create-home --home-dir /app yolo

COPY --from=builder /opt/venv /opt/venv

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
# Runtime — ROCm (AMD GPU)
############################
FROM rocm/dev-ubuntu-22.04:6.4 AS runtime-rocm

# Базовый образ уже содержит /opt/rocm с HIP-библиотеками ROCm 6.4
# (libamdhip64, libhiprtc, libMIOpen и т.д.). onnxruntime-rocm 1.21.0 собран
# под эту версию — soname совпадают (libhipblas.so.2 и т.п.).
# Доустанавливаем только то, чего нет в dev-образе, но нужно onnxruntime.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ACUITY_WORKER_CONFIG=/etc/acuity-yolo/configs/worker.yml \
    LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm/llvm/lib" \
    # RDNA3 (gfx1100) иногда определяется как unsupported — фиксируем явно.
    HSA_OVERRIDE_GFX_VERSION="11.0.0"

# onnxruntime-rocm требует hipblas/miopen/rocblas/rocfft — ставим из ROCm-репо,
# который уже настроен в базовом образе. Плюс python (в dev-образе его нет) и
# системные либы для opencv. tini для корректных сигналов.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
        libgl1 libglib2.0-0 ca-certificates tini \
        hipblas miopen-hip rocblas rocfft \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 yolo \
    && useradd --system --uid 10001 --gid yolo --no-create-home --home-dir /app yolo

COPY --from=builder /opt/venv /opt/venv

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
ARG DEVICE=cpu
FROM runtime-${DEVICE} AS runtime
