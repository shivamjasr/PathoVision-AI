from __future__ import annotations

from pathlib import Path

import boto3
from botocore.client import Config

from backend.app.config import (
    OBJECT_STORE_ACCESS_KEY,
    OBJECT_STORE_BUCKET,
    OBJECT_STORE_ENDPOINT,
    OBJECT_STORE_ENABLED,
    OBJECT_STORE_REGION,
    OBJECT_STORE_SECRET_KEY,
    PUBLIC_ARTIFACT_NAMES,
)
from backend.app.config import UPLOAD_DIR


class ObjectStore:
    def __init__(self):
        self.enabled = OBJECT_STORE_ENABLED
        self.bucket = OBJECT_STORE_BUCKET
        self.client = None

        if self.enabled:
            self.client = boto3.client(
                "s3",
                endpoint_url=OBJECT_STORE_ENDPOINT or None,
                region_name=OBJECT_STORE_REGION,
                aws_access_key_id=OBJECT_STORE_ACCESS_KEY or None,
                aws_secret_access_key=OBJECT_STORE_SECRET_KEY or None,
                config=Config(signature_version="s3v4"),
            )

    def ping(self) -> bool:
        if not self.enabled or self.client is None:
            return True
        try:
            self.ensure_bucket()
            return True
        except Exception:
            return False

    def ensure_bucket(self) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return
        except Exception:
            pass
        self.client.create_bucket(Bucket=self.bucket)

    def upload_job_public(self, job_id: str) -> list[str]:
        if not self.enabled or self.client is None:
            return []

        self.ensure_bucket()
        local_dir = UPLOAD_DIR / "analysis" / job_id / "public"
        uploaded: list[str] = []

        for filename in PUBLIC_ARTIFACT_NAMES:
            path = local_dir / filename
            if not path.exists():
                continue
            key = f"analysis/{job_id}/{filename}"
            self.client.upload_file(str(path), self.bucket, key)
            uploaded.append(key)

        return uploaded


object_store = ObjectStore()
