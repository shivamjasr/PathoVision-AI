from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


JobStatus = Literal["queued", "running", "completed", "failed"]


class HealthResponse(BaseModel):
    status: str
    classifier_available: bool
    segmenter_available: bool
    database: bool
    broker: bool
    redis: bool
    object_storage: bool
    auth_required: bool


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int


class UserResponse(BaseModel):
    username: str
    role: str


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    password: str = Field(min_length=8, max_length=128)


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    stage: str
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobHistoryItem(BaseModel):
    job_id: str
    filename: str
    mode: str
    status: JobStatus
    progress: int
    stage: str
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class AuditEventResponse(BaseModel):
    id: int
    event_type: str
    message: str
    details: Dict[str, Any] = {}
    created_at: Optional[str] = None


class UploadResponse(BaseModel):
    job_id: str
    status: JobStatus
    filename: str


class ArtifactListResponse(BaseModel):
    job_id: str
    artifacts: list[str]
    object_storage: bool
