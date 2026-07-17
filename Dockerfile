# syntax=docker/dockerfile:1.6
# Dockerfile — образ inference-воркера acuity-yolo.
#
# Структура (упрощена из multi-stage: убраны builder-стадии):
#   1. exporter          — отдельная стадия только для экспорта YOLO→ONNX
#                          (ultralytics/torch тяжёлые, не нужны в рантайме).
#   2. runtime-{cpu|gpu-amd} — финальный образ, выбирается через FROM runtime-${DEVICE}.
#
# Почему без builder-стадий: перенос venv между базовыми образами (python:slim ↔
# rocm/dev) нестабилен — venv хранит симлинки на интерпретатор. Для ROCm это уже
# породило отдельный builder-gpu-amd на той же базе + каскад проблем с ARG в FROM.
# Для CPU multi-stage тоже почти не даёт выгоды (образ <300МБ и так). Поэтому
# venv создаётся прямо в runtime-стадии — проще, короче, меньше точек отказа.
#
# DEVICE=cpu     → python:3.11-slim + onnxruntime (PyPI, cp311).
# DEVICE=gpu-amd → rocm/dev-ubuntu-22.04:6.4-complete (полный ROCm 6.4, все
#                  HIP-библиотеки и symlink-ферма внутри — libhipblas.so.2 и т.п.
#                  на месте). onnxruntime-rocm 1.21.0 (cp310) с repo.radeon.com
#                  под системный Python 3.10 Ubuntu 22.04.
# Задел: DEVICE=gpu-nvidia — через nvidia/cuda-образы + onnxruntime-gpu.
#
# EP в рантайме задаётся EXECUTION_PROVIDER и должно совпадать с DEVICE.

# Глобальный ARG — виден во всех stages, включая FROM runtime-${DEVICE}.
ARG DEVICE=cpu

############################
# Exporter: YOLOv12m.pt → ONNX (опционально, отдельная стадия)
# ultralytics/torch нужны ТОЛЬКО здесь; в runtime не попадают.
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
        python3-venv libgl1 libglib2.0-0 ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv \
    && groupadd --system --gid 10001 yolo \
    && useradd --system --uid 10001 --gid yolo --no-create-home --home-dir /app yolo

COPY requirements.txt ./
RUN /opt/venv/bin/pip install --upgrade pip && /opt/venv/bin/pip install -r requirements.txt

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
# Runtime — AMD GPU (ROCm), одностадийный
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
    HSA_OVERRIDE_GFX_VERSION="11.0.0" \
    # MIOpen пишет кеши ядер в несколько каталогов (~/.config/miopen,
    # ~/.cache/miopen, system-db). Под пользователем yolo без HOME это падает
    # с Permission denied → find падает → Conv-узел падает. Решение: задаём
    # HOME=/tmp (доступен всем) И редиректим ВСЕ пути кеша MIOpen в /tmp.
    # ВНИМАНИЕ: НЕ задаём MIOPEN_FIND_MODE=1 (Fast) — в этом режиме MIOpen
    # полагается на кешированные бинарники ядер; для gfx1100 их нет, он
    # выбирает несуществующий алгоритм → "No invoker registered for conv".
    # Режим по умолчанию (Normal, 3=Hybrid) компилирует ядро сам при первом
    # запуске — медленнее на cold start, но работает.
    HOME="/tmp" \
    MIOPEN_USER_DB_PATH="/tmp/miopen-cache" \
    MIOPEN_SYSTEM_DB_PATH="/tmp/miopen-cache" \
    MIOPEN_CACHE_DIR="/tmp/miopen-cache"

# Системный Python 3.10 (Ubuntu 22.04). venv-модуль в отдельном пакете python3.10-venv.
# opencv-headless требует libgl1/libglib. tini для корректных сигналов.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3.10-venv python3-dev build-essential \
        libgl1 libglib2.0-0 ca-certificates tini \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && groupadd --system --gid 10001 yolo \
    && useradd --system --uid 10001 --gid yolo --no-create-home --home-dir /app yolo \
    # Каталог кеша MIOpen (MIOPEN_USER_DB_PATH=/tmp/miopen-cache) под пользователем
    # yolo, иначе MIOpen падает с Permission denied при поиске алгоритмов Conv.
    && mkdir -p /tmp/miopen-cache && chown -R yolo:yolo /tmp/miopen-cache

COPY requirements-gpu-amd.txt scripts/clear_execstack.py /tmp/build/
RUN /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r /tmp/build/requirements-gpu-amd.txt \
    # onnxruntime .so имеет PT_GNU_STACK=RWE (executable stack). Docker seccomp
    # режет mprotect(PROT_EXEC) → ImportError. Снимаем X-бит через Python
    # (execstack/prelink убраны из Debian). https://github.com/microsoft/onnxruntime/issues/24911
    && /opt/venv/bin/python /tmp/build/clear_execstack.py \
    && rm -rf /tmp/build

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
