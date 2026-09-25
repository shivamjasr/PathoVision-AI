from __future__ import annotations

import json
from typing import Any

import redis.asyncio as redis

from backend.app.config import REDIS_URL, REDIS_CACHE_TTL_SECONDS


_client = redis.from_url(
    REDIS_URL,
    decode_responses=True,
)


async def redis_ping() -> bool:
    try:
        return bool(await _client.ping())
    except Exception:
        return False


async def cache_get(key: str) -> Any | None:
    try:
        value = await _client.get(key)
        if value is None:
            return None
        return json.loads(value)
    except Exception:
        return None


async def cache_set(key: str, value: Any, ttl: int | None = None) -> bool:
    try:
        await _client.set(
            key,
            json.dumps(value, default=str),
            ex=ttl or REDIS_CACHE_TTL_SECONDS,
        )
        return True
    except Exception:
        return False


async def cache_delete(*keys: str) -> None:
    if not keys:
        return
    try:
        await _client.delete(*keys)
    except Exception:
        return


async def idempotency_get(key: str) -> str | None:
    try:
        return await _client.get(f"pathovision:idem:{key}")
    except Exception:
        return None


async def idempotency_reserve(key: str, value: str, ttl: int) -> bool:
    try:
        return bool(
            await _client.set(
                f"pathovision:idem:{key}",
                value,
                ex=ttl,
                nx=True,
            )
        )
    except Exception:
        # Idempotency becomes best-effort if Redis is temporarily down.
        return True


async def idempotency_finalize(key: str, job_id: str, ttl: int) -> None:
    try:
        await _client.set(
            f"pathovision:idem:{key}",
            job_id,
            ex=ttl,
        )
    except Exception:
        return


async def close_cache() -> None:
    await _client.aclose()
