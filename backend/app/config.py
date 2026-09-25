from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STORAGE_DIR = Path(os.getenv("PATHOVISION_STORAGE", str(PROJECT_ROOT / "runtime")))
UPLOAD_DIR = STORAGE_DIR / "uploads"
JOB_DIR = STORAGE_DIR / "jobs"

MAX_UPLOAD_GB = float(os.getenv("PATHOVISION_MAX_UPLOAD_GB", "8"))
API_HOST = os.getenv("PATHOVISION_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PATHOVISION_API_PORT", "8000"))
FRONTEND_ORIGIN = os.getenv("PATHOVISION_FRONTEND_ORIGIN", "http://localhost:5173")

DATABASE_URL = os.getenv(
    "PATHOVISION_DATABASE_URL",
    "postgresql+psycopg://pathovision:pathovision@localhost:5432/pathovision",
)
CELERY_BROKER_URL = os.getenv(
    "PATHOVISION_CELERY_BROKER_URL",
    "amqp://pathovision:pathovision@localhost:5672/pathovision",
)
CELERY_QUEUE = os.getenv("PATHOVISION_CELERY_QUEUE", "analysis")

REDIS_URL = os.getenv("PATHOVISION_REDIS_URL", "redis://localhost:6379/0")
REDIS_CACHE_TTL_SECONDS = int(os.getenv("PATHOVISION_REDIS_CACHE_TTL", "3"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("PATHOVISION_RATE_LIMIT_PER_MINUTE", "120"))
ANALYZE_RATE_LIMIT_PER_MINUTE = int(os.getenv("PATHOVISION_ANALYZE_RATE_LIMIT_PER_MINUTE", "10"))
LOGIN_RATE_LIMIT_PER_MINUTE = int(os.getenv("PATHOVISION_LOGIN_RATE_LIMIT_PER_MINUTE", "12"))
REGISTER_RATE_LIMIT_PER_MINUTE = int(os.getenv("PATHOVISION_REGISTER_RATE_LIMIT_PER_MINUTE", "6"))
IDEMPOTENCY_TTL_SECONDS = int(os.getenv("PATHOVISION_IDEMPOTENCY_TTL_SECONDS", "86400"))

AUTH_REQUIRED = os.getenv("PATHOVISION_AUTH_REQUIRED", "false").lower() in {"1", "true", "yes"}
JWT_SECRET = os.getenv("PATHOVISION_JWT_SECRET", "dev-only-change-me-pathovision")
JWT_ALGORITHM = os.getenv("PATHOVISION_JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("PATHOVISION_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
BOOTSTRAP_USERNAME = os.getenv("PATHOVISION_BOOTSTRAP_USERNAME", "")
BOOTSTRAP_PASSWORD = os.getenv("PATHOVISION_BOOTSTRAP_PASSWORD", "")

CLASSIFIER_CHECKPOINT = Path(
    os.getenv(
        "PATHOVISION_CLASSIFIER_CHECKPOINT",
        str(PROJECT_ROOT / "checkpoints" / "best_resnet18.pt"),
    )
)
UNET_CHECKPOINT = Path(
    os.getenv(
        "PATHOVISION_UNET_CHECKPOINT",
        str(PROJECT_ROOT / "checkpoints" / "best_unet.pt"),
    )
)
DEFAULT_TILE_SIZE = int(os.getenv("PATHOVISION_TILE_SIZE", "256"))
DEFAULT_MIN_TISSUE = float(os.getenv("PATHOVISION_MIN_TISSUE", "0.25"))
DEFAULT_CLASSIFIER_BATCH = int(os.getenv("PATHOVISION_CLASSIFIER_BATCH", "32"))
DEFAULT_SEGMENTATION_BATCH = int(os.getenv("PATHOVISION_SEGMENTATION_BATCH", "4"))

OBJECT_STORE_ENABLED = os.getenv("PATHOVISION_OBJECT_STORE_ENABLED", "false").lower() in {"1", "true", "yes"}
OBJECT_STORE_ENDPOINT = os.getenv("PATHOVISION_OBJECT_STORE_ENDPOINT", "")
OBJECT_STORE_BUCKET = os.getenv("PATHOVISION_OBJECT_STORE_BUCKET", "pathovision-artifacts")
OBJECT_STORE_REGION = os.getenv("PATHOVISION_OBJECT_STORE_REGION", "us-east-1")
OBJECT_STORE_ACCESS_KEY = os.getenv("PATHOVISION_OBJECT_STORE_ACCESS_KEY", "")
OBJECT_STORE_SECRET_KEY = os.getenv("PATHOVISION_OBJECT_STORE_SECRET_KEY", "")
PUBLIC_ARTIFACT_NAMES = (
    "thumbnail.jpg",
    "tumor_probability_heatmap.png",
    "tumor_heatmap_overlay.png",
    "tile_predictions.csv",
    "summary.json",
    "raw_stitched_mask.png",
    "cleaned_tumor_mask.png",
    "tumor_segmentation_overlay.png",
    "tumor_regions.csv",
    "tumor_quantification.json",
)


def ensure_storage_dirs():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
