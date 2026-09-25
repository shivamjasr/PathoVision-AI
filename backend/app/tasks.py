from pathlib import Path
import logging

from backend.app.celery_app import celery_app
from backend.app.config import DEFAULT_MIN_TISSUE, DEFAULT_TILE_SIZE
from backend.app.db.session import SessionLocal
from backend.app.db.store import DatabaseJobStore
from backend.app.metrics import JOBS_COMPLETED_TOTAL, JOBS_FAILED_TOTAL
from backend.app.object_store import object_store
from backend.app.pipeline import run_analysis


logger = logging.getLogger("pathovision.worker")


def get_job_store():
    return DatabaseJobStore(SessionLocal)


@celery_app.task(
    bind=True,
    name="pathovision.run_analysis",
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=3,
    rate_limit="4/m",
)
def run_analysis_task(
    self,
    job_id: str,
    slide_path: str,
    mode: str = "classify",
    tile_size: int = DEFAULT_TILE_SIZE,
    min_tissue: float = DEFAULT_MIN_TISSUE,
):
    store = get_job_store()
    record = store.get(job_id)
    job_mode = record.get("mode", mode) if record else mode

    store.update(
        job_id,
        status="running",
        progress=1,
        stage="worker_started",
        message=f"Analysis worker started (attempt {self.request.retries + 1}).",
    )

    try:
        result = run_analysis(
            job=job_id,
            job_store=store,
            slide_path=Path(slide_path),
            mode=mode,
            tile_size=tile_size,
            min_tissue=min_tissue,
        )

        if object_store.enabled:
            try:
                uploaded = object_store.upload_job_public(job_id)
                store.update(
                    job_id,
                    message=f"Analysis completed. {len(uploaded)} artifacts mirrored to object storage.",
                )
            except Exception as storage_exc:
                logger.warning(
                    "Object storage mirror failed: %s",
                    storage_exc,
                    extra={"request_id": job_id},
                )
                store.update(
                    job_id,
                    message="Analysis completed; object-storage mirror failed, local artifacts retained.",
                )

        JOBS_COMPLETED_TOTAL.labels(job_mode).inc()
        return result

    except Exception as exc:
        # Only permanent task-level failures are finalized here. Transport-related
        # ConnectionError/TimeoutError exceptions are eligible for Celery retry.
        if isinstance(exc, (ConnectionError, TimeoutError)) and self.request.retries < 3:
            store.update(
                job_id,
                status="queued",
                progress=0,
                stage="retry_scheduled",
                message=f"Transient failure; retry scheduled (attempt {self.request.retries + 2}).",
                error=str(exc),
            )
        else:
            store.update(
                job_id,
                status="failed",
                progress=100,
                stage="failed",
                message="Analysis worker failed.",
                error=str(exc),
            )
            JOBS_FAILED_TOTAL.labels(job_mode).inc()

        raise
