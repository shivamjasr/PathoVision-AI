from __future__ import annotations

import time

from fastapi import Request
import redis.asyncio as redis

from backend.app.config import REDIS_URL


_client = redis.from_url(REDIS_URL, decode_responses=True)


async def check_rate_limit(
    request: Request,
    limit: int,
    window_seconds: int,
) -> tuple[bool, int, int]:
    """Fixed-window limiter backed by Redis; fails open if Redis is down."""
    host = request.client.host if request.client else "unknown"
    bucket = int(time.time() // window_seconds)
    key = f"pathovision:rl:{host}:{request.url.path}:{bucket}"

    try:
        count = int(await _client.incr(key))
        if count == 1:
            await _client.expire(key, window_seconds)
        remaining = max(0, limit - count)
        return count <= limit, remaining, count
    except Exception:
        return True, limit, 0


async def close_rate_limiter() -> None:
    await _client.aclose()
