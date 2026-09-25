from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.config import DATABASE_URL


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    expire_on_commit=False,
)


def get_session():
    with SessionLocal() as session:
        yield session
