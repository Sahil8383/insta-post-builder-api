"""SQLAlchemy engine and session factory."""

import logging
from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


class Base(DeclarativeBase):
    pass


from app.models import Post


def _ensure_post_columns_sqlite(engine) -> None:
    """Add ``user_query`` / ``session_summary`` if upgrading an existing DB."""
    insp = inspect(engine)
    if not insp.has_table("posts"):
        return
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(posts)")).fetchall()
        col_names = {str(row[1]) for row in rows}
        if "user_query" not in col_names:
            conn.execute(text("ALTER TABLE posts ADD COLUMN user_query TEXT DEFAULT ''"))
        if "session_summary" not in col_names:
            conn.execute(
                text("ALTER TABLE posts ADD COLUMN session_summary TEXT DEFAULT ''")
            )


def _ensure_pgvector_extension(engine) -> None:
    """Enable pgvector when using PostgreSQL (no-op if already present or disallowed)."""
    if engine.dialect.name != "postgresql":
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as exc:
        logger.warning(
            "Could not enable pgvector extension (add columns later or grant EXTENSION): %s",
            exc,
        )


def ensure_posts_schema() -> None:
    """Create ``posts`` table if missing; apply additive column upgrades."""

    engine = get_engine()
    _ensure_pgvector_extension(engine)
    insp = inspect(engine)
    if not insp.has_table("posts"):
        Base.metadata.create_all(bind=engine, tables=[Post.__table__])
    elif engine.dialect.name == "sqlite":
        _ensure_post_columns_sqlite(engine)


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        url = settings.resolved_database_url
        kwargs: dict = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        _engine = create_engine(url, **kwargs)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def get_session_factory():
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
