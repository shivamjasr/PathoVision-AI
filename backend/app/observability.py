from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.metrics import HTTP_LATENCY_SECONDS, HTTP_REQUESTS_TOTAL


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        return json.dumps(payload, separators=(",", ":"))


def configure_logging() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - started
            HTTP_REQUESTS_TOTAL.labels(
                request.method,
                request.url.path,
                "500",
            ).inc()
            HTTP_LATENCY_SECONDS.labels(
                request.method,
                request.url.path,
            ).observe(duration)
            logging.getLogger("pathovision.http").exception(
                "Unhandled request exception",
                extra={"request_id": request_id},
            )
            raise

        duration = time.perf_counter() - started
        status = str(response.status_code)
        HTTP_REQUESTS_TOTAL.labels(
            request.method,
            request.url.path,
            status,
        ).inc()
        HTTP_LATENCY_SECONDS.labels(
            request.method,
            request.url.path,
        ).observe(duration)

        response.headers["X-Request-ID"] = request_id
        return response
