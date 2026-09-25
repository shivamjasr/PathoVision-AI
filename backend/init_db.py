from sqlalchemy import text

from backend.app.auth.service import hash_password
from backend.app.config import BOOTSTRAP_PASSWORD, BOOTSTRAP_USERNAME
from backend.app.db.models import Base, User
from backend.app.db.session import SessionLocal, engine


def ensure_legacy_columns():
    statements = [
        "ALTER TABLE analysis_jobs ADD COLUMN IF NOT EXISTS owner_username VARCHAR(64)",
        "CREATE INDEX IF NOT EXISTS ix_analysis_jobs_owner_username ON analysis_jobs (owner_username)",
    ]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def bootstrap_admin():
    if not BOOTSTRAP_USERNAME or not BOOTSTRAP_PASSWORD:
        return

    with SessionLocal() as session:
        existing = session.query(User).filter(User.username == BOOTSTRAP_USERNAME).first()
        if existing:
            return

        session.add(
            User(
                username=BOOTSTRAP_USERNAME,
                hashed_password=hash_password(BOOTSTRAP_PASSWORD),
                role="admin",
            )
        )
        session.commit()


def main():
    Base.metadata.create_all(bind=engine)
    ensure_legacy_columns()
    bootstrap_admin()
    print("PathoVision PostgreSQL schema initialized.")


if __name__ == "__main__":
    main()
