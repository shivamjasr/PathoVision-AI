from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash
from sqlalchemy import select

from backend.app.config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    AUTH_REQUIRED,
    JWT_ALGORITHM,
    JWT_SECRET,
)
from backend.app.db.models import User
from backend.app.db.session import SessionLocal


password_hash = PasswordHash.recommended()
DUMMY_HASH = password_hash.hash("pathovision-dummy-password")
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/token",
    auto_error=False,
)


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return password_hash.verify(password, hashed_password)


def create_access_token(username: str, role: str) -> str:
    expires = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload: dict[str, Any] = {
        "sub": username,
        "role": role,
        "exp": expires,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_user(username: str) -> User | None:
    with SessionLocal() as session:
        return session.scalar(
            select(User).where(User.username == username)
        )


def authenticate_user(username: str, password: str) -> User | None:
    user = get_user(username)
    if user is None:
        verify_password(password, DUMMY_HASH)
        return None

    if not verify_password(password, user.hashed_password):
        return None

    return user


def current_user_from_token(token: str) -> dict[str, str]:
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
        )
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    username = payload.get("sub")
    role = payload.get("role", "user")

    if not username or not isinstance(username, str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has no valid subject.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "username": username,
        "role": str(role),
    }


def require_user(token: str | None = Depends(oauth2_scheme)) -> dict[str, str]:
    if not AUTH_REQUIRED:
        return {
            "username": "local-dev",
            "role": "admin",
        }

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    identity = current_user_from_token(token)
    user = get_user(identity["username"])

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return identity


def require_admin(user: dict[str, str] = Depends(require_user)) -> dict[str, str]:
    if user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator role required.",
        )
    return user
