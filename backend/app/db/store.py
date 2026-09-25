from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from backend.app.db.models import AnalysisJob, AuditEvent


class DatabaseJobStore:
    """Persistent job store shared by the API and Celery workers."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def create(self, job_id: str, filename: str, file_path: Path, mode: str, owner_username: str | None = None) -> dict:
        with self.session_factory() as session:
            job = AnalysisJob(
                job_id=job_id,
                filename=filename,
                file_path=str(file_path),
                mode=mode,
                owner_username=owner_username,
                status="queued",
                progress=0,
                stage="queued",
                message="Waiting for an analysis worker.",
            )
            session.add(job)
            session.add(
                AuditEvent(
                    job_id=job_id,
                    event_type="created",
                    message="Analysis job created.",
                    details={"mode": mode, "filename": filename, "owner": owner_username},
                )
            )
            session.commit()
            return self._to_dict(job)

    def get(self, job_id: str, owner_username: str | None = None):
        with self.session_factory() as session:
            job = session.get(AnalysisJob, job_id)
            if job is None:
                return None
            if owner_username is not None and job.owner_username not in {None, owner_username}:
                return None
            return self._to_dict(job)

    def list_recent(self, limit=50, owner_username: str | None = None):
        with self.session_factory() as session:
            statement = select(AnalysisJob)
            if owner_username is not None:
                statement = statement.where(AnalysisJob.owner_username == owner_username)
            statement = statement.order_by(desc(AnalysisJob.created_at)).limit(limit)
            rows = list(session.scalars(statement))
            return [self._to_dict(job) for job in rows]

    def update(self, job_id: str, **changes: Any):
        with self.session_factory() as session:
            job = session.get(AnalysisJob, job_id)
            if job is None:
                raise KeyError(f"Unknown job: {job_id}")

            audit_fields = {}
            for key, value in changes.items():
                if not hasattr(job, key):
                    raise ValueError(f"Unknown AnalysisJob field: {key}")
                setattr(job, key, value)
                if key in {"status", "progress", "stage", "message"}:
                    audit_fields[key] = value

            if changes.get("status") in {"completed", "failed"}:
                job.completed_at = datetime.now(timezone.utc)

            if audit_fields:
                session.add(
                    AuditEvent(
                        job_id=job_id,
                        event_type="state_change",
                        message=changes.get("message", "Job state updated."),
                        details=audit_fields,
                    )
                )

            session.commit()
            return self._to_dict(job)

    def events(self, job_id: str):
        with self.session_factory() as session:
            statement = select(AuditEvent).where(AuditEvent.job_id == job_id).order_by(AuditEvent.created_at)
            rows = list(session.scalars(statement))
            return [
                {
                    "id": row.id,
                    "event_type": row.event_type,
                    "message": row.message,
                    "details": row.details or {},
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

    @staticmethod
    def _to_dict(job: AnalysisJob) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "filename": job.filename,
            "file_path": job.file_path,
            "mode": job.mode,
            "owner_username": job.owner_username,
            "status": job.status,
            "progress": job.progress,
            "stage": job.stage,
            "message": job.message,
            "error": job.error,
            "result": job.result,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "updated_at": job.updated_at.isoformat() if job.updated_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }
