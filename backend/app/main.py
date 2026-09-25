from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import shutil
import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.security import OAuth2PasswordRequestForm
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from backend.app.auth.service import (
    authenticate_user,
    create_access_token,
    get_user,
    hash_password,
    require_admin,
    require_user,
)
from backend.app.cache import cache_delete, cache_get, cache_set, idempotency_finalize, idempotency_get, idempotency_reserve
from backend.app.celery_app import celery_app
from backend.app.config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ANALYZE_RATE_LIMIT_PER_MINUTE,
    AUTH_REQUIRED,
    CLASSIFIER_CHECKPOINT,
    DEFAULT_MIN_TISSUE,
    DEFAULT_TILE_SIZE,
    FRONTEND_ORIGIN,
    IDEMPOTENCY_TTL_SECONDS,
    LOGIN_RATE_LIMIT_PER_MINUTE,
    MAX_UPLOAD_GB,
    OBJECT_STORE_BUCKET,
    OBJECT_STORE_ENABLED,
    RATE_LIMIT_PER_MINUTE,
    REDIS_URL,
    REGISTER_RATE_LIMIT_PER_MINUTE,
    UNET_CHECKPOINT,
    UPLOAD_DIR,
    ensure_storage_dirs,
)
from backend.app.db.models import Base, User
from backend.app.db.session import SessionLocal, engine
from backend.app.db.store import DatabaseJobStore
from backend.app.metrics import JOBS_SUBMITTED_TOTAL
from backend.app.object_store import object_store
from backend.app.observability import RequestLoggingMiddleware, configure_logging
from backend.app.rate_limit import check_rate_limit
from backend.app.security_utils import safe_filename
from backend.app.schemas import (
    ArtifactListResponse,
    AuditEventResponse,
    HealthResponse,
    JobHistoryItem,
    JobResponse,
    RegisterRequest,
    TokenResponse,
    UploadResponse,
    UserResponse,
)


configure_logging()
logger = logging.getLogger("pathovision.api")
ensure_storage_dirs()
Base.metadata.create_all(bind=engine)
try:
    from backend.init_db import ensure_legacy_columns
    ensure_legacy_columns()
except Exception:
    # Direct `uvicorn` startup against a brand-new DB is still handled by create_all.
    pass

app = FastAPI(
    title="PathoVision AI API",
    description=(
        "Whole-slide pathology analysis with FastAPI, PostgreSQL, RabbitMQ, "
        "Celery, Redis, and optional S3-compatible object storage."
    ),
    version="1.0.0",
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = DatabaseJobStore(SessionLocal)

ALLOWED_SUFFIXES = {
    ".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn",
    ".png", ".jpg", ".jpeg",
}


def artifact_public_dir(job_id: str) -> Path:
    return UPLOAD_DIR / "analysis" / job_id / "public"


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if not path.startswith("/api/") or path in {
        "/api/health",
        "/api/health/live",
        "/api/health/ready",
    }:
        return await call_next(request)

    if path == "/api/auth/token":
        limit = LOGIN_RATE_LIMIT_PER_MINUTE
    elif path == "/api/auth/register":
        limit = REGISTER_RATE_LIMIT_PER_MINUTE
    elif path == "/api/analyze":
        limit = ANALYZE_RATE_LIMIT_PER_MINUTE
    else:
        limit = RATE_LIMIT_PER_MINUTE

    allowed, remaining, count = await check_rate_limit(request, limit, 60)
    if not allowed:
        response = Response(
            content='{"detail":"Rate limit exceeded. Please retry later."}',
            status_code=429,
            media_type="application/json",
        )
        response.headers["Retry-After"] = "60"
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = "0"
        return response

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
    response.headers["X-RateLimit-Used"] = str(count)
    return response


@app.get("/api/health/live")
def liveness():
    return {"status": "alive"}


@app.get("/api/health/ready", response_model=HealthResponse)
def readiness():
    return _health_payload()


@app.get("/api/health", response_model=HealthResponse)
def health():
    return _health_payload()


def _health_payload() -> dict:
    database_ok = False
    broker_ok = False
    redis_ok = False

    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
            database_ok = True
    except Exception:
        pass

    try:
        connection = celery_app.connection_for_read()
        connection.connect()
        connection.release()
        broker_ok = True
    except Exception:
        pass

    try:
        import redis
        client = redis.Redis.from_url(REDIS_URL)
        redis_ok = bool(client.ping())
        client.close()
    except Exception:
        pass

    object_ok = object_store.ping()
    return {
        "status": "ok" if database_ok and broker_ok and redis_ok else "degraded",
        "classifier_available": CLASSIFIER_CHECKPOINT.exists(),
        "segmenter_available": UNET_CHECKPOINT.exists(),
        "database": database_ok,
        "broker": broker_ok,
        "redis": redis_ok,
        "object_storage": object_ok,
        "auth_required": AUTH_REQUIRED,
    }


@app.post("/api/auth/register", response_model=UserResponse, status_code=201)
def register(payload: RegisterRequest):
    if get_user(payload.username):
        raise HTTPException(status_code=409, detail="Username already exists.")

    with SessionLocal() as session:
        session.add(
            User(
                username=payload.username,
                hashed_password=hash_password(payload.password),
                role="user",
            )
        )
        session.commit()

    return UserResponse(username=payload.username, role="user")


@app.post("/api/auth/token", response_model=TokenResponse)
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = authenticate_user(form.username, form.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenResponse(
        access_token=create_access_token(user.username, user.role),
        expires_in_seconds=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@app.get("/api/auth/me", response_model=UserResponse)
def me(user: dict[str, str] = Depends(require_user)):
    return UserResponse(username=user["username"], role=user["role"])


async def save_upload(upload: UploadFile, destination: Path):
    max_bytes = int(MAX_UPLOAD_GB * 1024 * 1024 * 1024)
    total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("wb") as file:
        while True:
            chunk = await upload.read(8 * 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds the configured {MAX_UPLOAD_GB:g} GB limit.",
                )
            file.write(chunk)
    await upload.close()



@app.post("/api/analyze", response_model=UploadResponse, status_code=202)
async def analyze(
    file: Annotated[UploadFile, File()],
    mode: Annotated[str, Query(pattern="^(classify|full)$")] = "classify",
    tile_size: Annotated[int, Query(ge=64, le=1024)] = DEFAULT_TILE_SIZE,
    min_tissue: Annotated[float, Query(ge=0, le=1)] = DEFAULT_MIN_TISSUE,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    user: dict[str, str] = Depends(require_user),
):
    filename = safe_filename(file.filename)
    idem_key = None

    if idempotency_key:
        if len(idempotency_key) > 128:
            raise HTTPException(status_code=400, detail="Idempotency-Key is too long.")
        raw_identity = f"{user['username']}:{idempotency_key}"
        idem_key = hashlib.sha256(raw_identity.encode("utf-8")).hexdigest()
        existing = await idempotency_get(idem_key)
        owner = None if user["role"] == "admin" else user["username"]

        if existing and existing not in {"PENDING", "FAILED"}:
            existing_job = jobs.get(existing, owner)
            if existing_job:
                return UploadResponse(
                    job_id=existing_job["job_id"],
                    status=existing_job["status"],
                    filename=existing_job["filename"],
                )
        elif existing == "PENDING":
            raise HTTPException(status_code=409, detail="An identical request is already being created.")

        if not await idempotency_reserve(idem_key, "PENDING", IDEMPOTENCY_TTL_SECONDS):
            raise HTTPException(status_code=409, detail="Idempotency-Key is already in use.")

    job_id = uuid.uuid4().hex
    slide_path = UPLOAD_DIR / job_id / filename

    try:
        try:
            await save_upload(file, slide_path)
        except Exception:
            shutil.rmtree(slide_path.parent, ignore_errors=True)
            raise

        jobs.create(
            job_id=job_id,
            filename=filename,
            file_path=slide_path,
            mode=mode,
            owner_username=user["username"],
        )
        jobs.update(job_id, progress=0, stage="queued", message="Submitted to the analysis queue.")

        try:
            celery_result = celery_app.send_task(
                "pathovision.run_analysis",
                args=[job_id, str(slide_path), mode, tile_size, min_tissue],
                queue="analysis",
            )
        except Exception as queue_exc:
            jobs.update(
                job_id,
                status="failed",
                progress=100,
                stage="queue_failed",
                message="Could not enqueue analysis.",
                error=str(queue_exc),
            )
            raise HTTPException(
                status_code=503,
                detail="Analysis broker is unavailable. The upload was saved but the job was not queued.",
            ) from queue_exc

        jobs.update(job_id, message=f"Queued for worker processing. Celery task={celery_result.id}.")
        if idem_key:
            await idempotency_finalize(idem_key, job_id, IDEMPOTENCY_TTL_SECONDS)
        await cache_delete(f"pathovision:jobs:{user['username']}:25")
        JOBS_SUBMITTED_TOTAL.labels(mode).inc()
        return UploadResponse(job_id=job_id, status="queued", filename=filename)

    except HTTPException:
        raise
    except Exception as exc:
        if idem_key:
            await idempotency_finalize(idem_key, "FAILED", 300)
        logger.exception("Analysis submission failed")
        raise HTTPException(
            status_code=503,
            detail="Analysis could not be submitted. Check API dependencies and retry.",
        ) from exc


@app.get("/api/jobs", response_model=list[JobHistoryItem])
async def list_jobs(
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    user: dict[str, str] = Depends(require_user),
):
    cache_key = f"pathovision:jobs:{user['username']}:{limit}"
    cached = await cache_get(cache_key)
    if cached is not None:
        return cached

    owner = None if user["role"] == "admin" else user["username"]
    rows = jobs.list_recent(limit=limit, owner_username=owner)
    payload = [
        JobHistoryItem(
            job_id=row["job_id"],
            filename=row["filename"],
            mode=row["mode"],
            status=row["status"],
            progress=row["progress"],
            stage=row["stage"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        ).model_dump()
        for row in rows
    ]
    await cache_set(cache_key, payload, ttl=3)
    return payload


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, user: dict[str, str] = Depends(require_user)):
    owner = None if user["role"] == "admin" else user["username"]
    cache_key = f"pathovision:job:{user['username']}:{job_id}"
    cached = await cache_get(cache_key)
    if cached is not None:
        return cached

    record = jobs.get(job_id, owner)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    payload = JobResponse(
        job_id=record["job_id"],
        status=record["status"],
        progress=record["progress"],
        stage=record["stage"],
        message=record["message"],
        result=record.get("result"),
        error=record.get("error"),
    ).model_dump()
    await cache_set(cache_key, payload, ttl=2)
    return payload


@app.get("/api/jobs/{job_id}/events", response_model=list[AuditEventResponse])
def get_job_events(job_id: str, user: dict[str, str] = Depends(require_user)):
    owner = None if user["role"] == "admin" else user["username"]
    if jobs.get(job_id, owner) is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return jobs.events(job_id)


@app.get("/api/jobs/{job_id}/artifacts", response_model=ArtifactListResponse)
def list_artifacts(job_id: str, user: dict[str, str] = Depends(require_user)):
    owner = None if user["role"] == "admin" else user["username"]
    if jobs.get(job_id, owner) is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    public_dir = artifact_public_dir(job_id)
    names = []
    if public_dir.exists():
        names = sorted(item.name for item in public_dir.iterdir() if item.is_file())
    return ArtifactListResponse(job_id=job_id, artifacts=names, object_storage=OBJECT_STORE_ENABLED)


@app.get("/api/jobs/{job_id}/files/{filename}")
def get_artifact(
    job_id: str,
    filename: str,
    user: dict[str, str] = Depends(require_user),
):
    owner = None if user["role"] == "admin" else user["username"]
    if jobs.get(job_id, owner) is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    public_dir = artifact_public_dir(job_id)
    candidate = public_dir / Path(filename).name
    try:
        resolved = candidate.resolve()
        root = public_dir.resolve()
        resolved.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid artifact path.") from exc

    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(path=resolved)


@app.get("/api/admin/object-store")
def object_store_status(_: dict[str, str] = Depends(require_admin)):
    return {
        "enabled": object_store.enabled,
        "bucket": OBJECT_STORE_BUCKET,
        "healthy": object_store.ping(),
    }


@app.get("/api/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/docs-info")
def docs_info(user: dict[str, str] = Depends(require_user)):
    return {
        "version": "1.0.0",
        "frontend": FRONTEND_ORIGIN,
        "allowed_upload_suffixes": sorted(ALLOWED_SUFFIXES),
        "max_upload_gb": MAX_UPLOAD_GB,
        "database": "PostgreSQL",
        "broker": "RabbitMQ",
        "worker": "Celery",
        "cache_rate_limit": "Redis",
        "object_storage": "S3-compatible" if OBJECT_STORE_ENABLED else "disabled",
        "auth_required": AUTH_REQUIRED,
        "features": [
            "JWT authentication",
            "Redis rate limiting and short-lived job cache",
            "Idempotency-Key protected job submission",
            "Celery retry/backoff",
            "structured request logging",
            "Prometheus metrics",
            "optional S3-compatible artifact mirroring",
            "classification heatmap",
            "full workflow with U-Net segmentation and tumor quantification",
        ],
        "notes": [
            "PostgreSQL remains the application source of truth for job state and audit events.",
            "U-Net full workflow is inference-only; it does not replace annotated training/evaluation splits.",
            "All area values from the segmentation stage remain thumbnail-pixel units unless physical slide calibration is added.",
        ],
        "current_user": user["username"],
    }
