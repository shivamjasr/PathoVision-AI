# PathoVision AI

PathoVision AI is an end-to-end digital pathology research/engineering project for high-resolution whole-slide image analysis.

It combines patch-level deep learning, whole-slide processing, tumor localization, pixel-level segmentation, quantitative region analysis, classical ML + deep-feature fusion, experiment analysis, and a full-stack application.

## What the project contains

```text
PCam patches / annotated pathology patches
        │
        ├── baseline CNN
        ├── ResNet18 transfer learning
        ├── handcrafted + deep feature hybrid models
        └── evaluation / Grad-CAM / error analysis

Whole-slide image (WSI)
        │
        ▼
OpenSlide reader
        │
        ▼
Low-resolution thumbnail + tissue mask
        │
        ▼
Tissue-aware tile extraction
        │
        ▼
ResNet18 tile classification
        │
        ▼
Coordinate-aware probability heatmap
        │
        ├──────────────► classification results
        │
        ▼
U-Net tile segmentation
        │
        ▼
Slide-level mask stitching + morphology
        │
        ▼
Connected-component tumor regions
        │
        ▼
Quantification

React UI
   │
FastAPI
   │
PostgreSQL ── job history + audit trail
   │
RabbitMQ ──► Celery worker
   │
Redis ── cache + rate limiting + idempotency
   │
MinIO ── optional S3-compatible artifact mirror
   │
Flower ── Celery monitoring
```

## Repository layout

```text
PathoVision-AI/
├── backend/
│   ├── app/
│   │   ├── auth/                 # JWT + password hashing
│   │   ├── db/                   # PostgreSQL models/store
│   │   ├── cache.py              # Redis cache/idempotency helpers
│   │   ├── celery_app.py         # Celery/RabbitMQ configuration
│   │   ├── main.py               # FastAPI API
│   │   ├── metrics.py            # Prometheus metrics
│   │   ├── object_store.py       # S3-compatible artifact mirroring
│   │   ├── observability.py      # JSON logging + request IDs
│   │   ├── pipeline.py           # end-to-end WSI analysis
│   │   ├── rate_limit.py         # Redis rate limiter
│   │   ├── schemas.py            # API schemas
│   │   ├── security_utils.py     # upload validation
│   │   └── segmentation_pipeline.py
│   ├── init_db.py
│   ├── requirements.txt
│   └── tests/
├── frontend/
│   ├── src/
│   ├── Dockerfile
│   └── package.json
├── src/
│   ├── datasets/                 # PCam dataset handling
│   ├── experiments/              # lightweight experiment tracker
│   ├── hybrid/                   # handcrafted + deep features
│   ├── inference/                # WSI classifier inference + heatmaps
│   ├── models/                   # CNN / ResNet18 / U-Net
│   ├── segmentation/             # masks, inference, XML, quantification
│   └── wsi/                      # OpenSlide reader/tissue/tiling
├── train.py
├── train_resnet.py
├── train_unet.py
├── train_hybrid.py
├── evaluate.py
├── evaluate_resnet.py
├── evaluate_unet.py
├── infer_wsi.py
├── segment_wsi.py
├── extract_patches.py
├── extract_segmentation_patches.py
├── extract_hybrid_features.py
├── predict_image.py
├── predict_hybrid.py
├── collect_predictions.py
├── optimize_threshold.py
├── plot_evaluation_curves.py
├── ablation_study.py
├── feature_importance.py
├── error_analysis.py
├── gradcam_resnet.py
├── compare_models.py
├── compare_hybrid_models.py
├── inspect_dataset.py
├── inspect_wsi.py
├── smoke_test.py
├── dry_run_wsi.py
├── make_demo_slide.py
├── make_segmentation_demo.py
├── demo_segmentation.py
├── download_data.py
├── docker-compose.yml
├── start.sh
├── requirements.txt
└── .dockerignore
```

## Main ML components

### Patch classification

A compact CNN provides a baseline for 96×96 pathology patches. The main deep-learning classifier is ResNet18 transfer learning with ImageNet preprocessing.

The ResNet workflow supports three training strategies:

```bash
python train_resnet.py --strategy frozen
python train_resnet.py --strategy finetune
python train_resnet.py --strategy full
```

Evaluate the trained classifier with:

```bash
python evaluate_resnet.py
```

### WSI processing

The WSI layer uses OpenSlide to read slide metadata, build a low-resolution thumbnail, estimate tissue regions, preserve level-0 coordinates, and extract tissue-rich tiles.

Inspect a slide:

```bash
python inspect_wsi.py /path/to/slide.svs
```

Extract tiles:

```bash
python extract_patches.py \
  /path/to/slide.svs \
  --tile-size 256 \
  --min-tissue 0.25 \
  --max-tiles 100
```

A PNG/JPG can be used for development through the synthetic/demo helpers.

### Tumor localization

`infer_wsi.py` runs the trained ResNet18 over extracted WSI tiles and reconstructs a coordinate-aware probability map.

```bash
python infer_wsi.py \
  --slide /path/to/slide.svs \
  --metadata /path/to/tiles.csv \
  --thumbnail /path/to/thumbnail.jpg
```

The resulting heatmap is localization evidence, not a pixel-level segmentation mask.

### U-Net segmentation and quantification

For annotated training data, segmentation patches are extracted from polygon annotations, then a binary U-Net is trained with BCE + Dice loss.

Create the synthetic segmentation demo:

```bash
python make_segmentation_demo.py
python train_unet.py \
  --manifest data/segmentation_demo/manifest.csv \
  --epochs 3 \
  --image-size 128 \
  --batch-size 8 \
  --num-workers 0
```

Evaluate and visualize:

```bash
python evaluate_unet.py \
  --manifest data/segmentation_demo/manifest.csv \
  --checkpoint checkpoints/best_unet.pt

python visualize_unet.py \
  --manifest data/segmentation_demo/manifest.csv \
  --checkpoint checkpoints/best_unet.pt
```

For an annotated WSI, split by slide/patient before patch extraction so patches from the same slide do not leak across train/validation/test.

The full application can run `mode=full`, which reuses the extracted WSI tiles for U-Net inference, stitches the masks back to thumbnail coordinates, cleans the binary mask, extracts connected tumor regions, and reports thumbnail-grid quantities.

Physical area such as mm² requires valid slide calibration metadata; thumbnail-pixel quantities are not clinical area measurements.

### Hybrid ML

`extract_hybrid_features.py` builds handcrafted color/texture/morphology features plus a 512-dimensional ResNet18 representation.

```bash
python extract_hybrid_features.py \
  --max-train 5000 \
  --max-val 1000 \
  --max-test 1000

python train_hybrid.py --features data/hybrid/features --trees 300
python compare_hybrid_models.py
python predict_hybrid.py /path/to/image.png
```

The scaler for the deep-feature model is fitted on training data only.

### Evaluation and explainability

The repository also includes:

```text
collect_predictions.py        Save validation/test probabilities
optimize_threshold.py         Select threshold on validation data
plot_evaluation_curves.py     ROC / PR curves
ablation_study.py             Feature-group ablations
feature_importance.py         Random-forest feature importance
error_analysis.py             False-positive / false-negative review
                                                        
gradcam_resnet.py             ResNet18 Grad-CAM visualization
```

Keep the final test set held out until model selection is complete.

## Full application

The production-style local stack is defined in exactly one Compose file:

```bash
docker compose up --build
```

Or:

```bash
./start.sh
```

Services:

| Service | Purpose | Port |
|---|---|---:|
| frontend | React/Vite application served by Nginx | 5173 |
| backend | FastAPI API | 8000 |
| postgres | persistent jobs, users, audit events | 5432 |
| rabbitmq | Celery broker | 5672 / 15672 |
| redis | cache, rate limit, idempotency | 6379 |
| worker | WSI/ML background analysis | internal |
| flower | Celery monitoring | 5555 |
| minio | S3-compatible artifact storage | 9000 / 9001 |

Open:

```text
Frontend       http://localhost:5173
API docs       http://localhost:8000/docs
Health         http://localhost:8000/api/health
Flower         http://localhost:5555
RabbitMQ UI    http://localhost:15672
MinIO Console  http://localhost:9001
```

The Compose environment bootstraps a development administrator:

```text
username: admin
password: pathovision
```

Change those credentials and `PATHOVISION_JWT_SECRET` before using the stack outside local development.

## Application flow

```text
Browser
  │
  │ JWT + multipart upload
  ▼
FastAPI
  │
  ├── validates upload
  ├── checks rate limit
  ├── checks Idempotency-Key
  ├── stores job in PostgreSQL
  └── publishes Celery task
          │
          ▼
      RabbitMQ
          │
          ▼
       Celery worker
          │
          ├── OpenSlide
          ├── tissue detection
          ├── tile extraction
          ├── ResNet18
          ├── heatmap generation
          ├── optional U-Net segmentation
          └── tumor quantification
          │
          ├── PostgreSQL job result
          ├── local browser-safe artifacts
          └── optional MinIO mirror
```

PostgreSQL is the source of truth for job status and audit events. Redis is deliberately used for short-lived infrastructure concerns rather than replacing persistent job state.

## API surface

```text
GET  /api/health
GET  /api/health/live
GET  /api/health/ready
GET  /api/metrics
GET  /api/docs-info

POST /api/auth/register
POST /api/auth/token
GET  /api/auth/me

POST /api/analyze
GET  /api/jobs
GET  /api/jobs/{job_id}
GET  /api/jobs/{job_id}/events
GET  /api/jobs/{job_id}/artifacts
GET  /api/jobs/{job_id}/files/{filename}

GET  /api/admin/object-store
```

## Model checkpoints

Large trained weights are intentionally not committed to the repository.

For the application to produce real analysis results, place the trained files at:

```text
checkpoints/best_resnet18.pt
checkpoints/best_unet.pt
```

The UI can still start without them, but analysis jobs requiring the missing checkpoint will fail with an explicit job error.

## Data

PCam can be used for quick patch-classification development. Whole-slide and annotated segmentation experiments should use datasets whose current terms permit the intended use and redistribution. Do not assume that a dataset license permits public redistribution or commercial use.

## Important scope note

PathoVision AI is a research/engineering system. Its model outputs are not a clinically validated diagnosis and should not be treated as medical advice.

## Development checks

Run Python syntax checks:

```bash
python -m compileall backend src *.py
```

Run the available tests:

```bash
pytest backend/tests -q
```

The included GitHub Actions workflow runs the same Python validation path in CI.
