# acuity-yolo

Stateless REST-сервис детекции объектов (YOLOv12, ONNX Runtime) для
классификации событий движения в проекте **acuity**.

Воркер **изолирован**: единственный интерфейс — `POST /detect` (multipart
JPEG). Никаких Kafka/NATS/Postgres/S3. Оркестрацию (забор кадров из S3, запись
детекций в БД, правила тегирования) выполняет `inference-service` (Go) из
`acuity-server` — он же является HTTP-клиентом к этому воркеру.

> Контракт и архитектура — `acuity-server/docs/EVENT_CLASSIFICATION.md`.

## API

```
POST /detect  (multipart: image=<JPEG|PNG>)
→ 200 {
      "boxes": [
        {"cls":"person","score":0.92,"x":120,"y":80,"w":60,"h":140}
      ],
      "classes": ["person"],
      "inference_ms": 47,
      "model": "yolov12n",
      "ep": "cpu"
    }

GET /healthz   — liveness (200 всегда, пока процесс жив)
GET /readyz    — readiness (200 если модель загружена, иначе 503)
GET /          — краткая сводка состояния
```

Координаты боксов — пиксели исходного изображения: `(x, y)` — верхний-левый
угол, `(w, h)` — размер. `classes` — уникальные имена классов на кадре (для
быстрого тегирования в inference-service).

## Модель

- **YOLOv12n** (`input_size=640`, ONNX) — зафиксированное решение ТЗ §10.
- Классы COCO-80 (стартовый набор: `person`, `car/truck/bus/motorcycle`,
  `suitcase/handbag/backpack`).
- Execution Provider — конфигом: `cpu` (по умолчанию) / `cuda` / `rocm` /
  `openvino`. Переключение — **без смены модели** (vendor-neutral).

### Экспорт модели

```bash
pip install ultralytics
python export_model.py                        # → models/yolov12n.onnx
python export_model.py --model yolov8n.pt      # альтернативная модель
```

В Docker-сборке экспорт выполняется автоматически (build-arg `EXPORT_MODEL=1`).
Если интернета нет или своя модель — соберите с `--build-arg EXPORT_MODEL=0`
и примонтируйте `.onnx` через volume (см. `docker-compose.yml`).

## Запуск

### Docker Compose (рекомендуется)

```bash
# Сборка с экспортом модели (нужен интернет для первой загрузки весей):
docker compose up -d --build

# Без экспорта модели (модель монтируется volume):
EXPORT_MODEL=0 docker compose up -d --build

# Проверка:
curl -s http://localhost:9104/healthz
curl -s -F "image=@test.jpg" http://localhost:9104/detect | jq
```

### Локально (для разработки)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install ultralytics && python export_model.py    # один раз
MODEL_PATH=models/yolov12n.onnx python -m uvicorn app.main:app --port 8000
```

## Масштабирование

Воркер **stateless**: модель в памяти процесса, каждый `/detect` независим.
Несколько способов горизонтального масштабирования (воркер может быть на
**отдельном сервере** в нескольких экземплярах):

1. **Несколько реплик за nginx round-robin** (для отдельного сервера).
   Раскомментируйте сервис `nginx` в `docker-compose.yml`, поднимайте реплики:
   `docker compose up --scale inference-worker=3`. Nginx балансирует по
   DNS-имени сервиса (Docker internal DNS).

2. **За HAProxy / Cloudflare / любым L7-балансером**: выставить
   `WORKER_HTTP_PORT` (по умолчанию 9104) наружу и завести пул инстансов.

3. **Кросс-сервер**: запустить `docker-compose.yml` на каждом сервере, указать
   в `inference-service` (`acuity-server`) балансер/round-robin DNS адрес как
   `INFERENCE_WORKER_URL`.

Inference-service (`acuity-server`) — единственный клиент воркера; он ходит по
HTTP и не требует от воркера обратных вызовов.

### Ресурсы (ориентировочно, ТЗ §5.3)

- CPU, YOLOv12n, 640px: ~50–150 мс/кадр. При интервале 3 с и 1 камере — копейки.
- AMD GPU + ROCm EP: ускорение ~5–15×. Переключение EP — конфигом, без пересборки.
- Для CPU-инстанса рекомендуется ~2 GiB RAM и ≥1 ядро на реплику.

## Конфигурация

Все параметры — env или `configs/worker.yml` (env имеет приоритет):

| Параметр | По умолчанию | Описание |
|---|---|---|
| `MODEL_PATH` | `models/yolov12n.onnx` | путь к `.onnx` |
| `EXECUTION_PROVIDER` | `cpu` | `cpu\|cuda\|rocm\|openvino\|tensorrt` |
| `CONF_THRESHOLD` | `0.4` | порог уверенности бокса |
| `IOU_THRESHOLD` | `0.5` | порог IoU для NMS |
| `INPUT_SIZE` | `640` | размер входа модели |
| `MODEL_NAME` | `yolov12n` | метка модели в ответе |
| `WORKERS` | `1` | uvicorn-воркеров (для нагрузки — `--scale` реплик) |
| `LOG_LEVEL` | `info` | `debug\|info\|warn\|error` |
| `MAX_IMAGE_BYTES` | `20971520` | лимит размера кадра (байт) |

## Структура

```
app/
  main.py          FastAPI: /detect, /healthz, /readyz, /
  model.py         ONNX Runtime session, letterbox, decode, NMS
  config.py        pydantic-settings (env + worker.yml)
  schemas.py       DetectResponse / Box
  logging_setup.py JSON-логи в stderr
export_model.py    экспорт YOLO → ONNX (ultralytics)
configs/worker.yml конфигурация
Dockerfile         multi-stage: builder → exporter (YOLO→ONNX) → runtime
docker-compose.yml автономный запуск + опц. redis/nginx
models/            сюда кладётся/монтируется .onnx (в git не коммитится)
```
