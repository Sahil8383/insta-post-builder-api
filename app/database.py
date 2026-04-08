"""SQLAlchemy engine and session factory."""

from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

_engine = None
_SessionLocal = None


class Base(DeclarativeBase):
    pass


def _migrate_usage_off_post_row(engine) -> None:
    """Move usage_* from post_generations into post_generation_usage; drop legacy columns."""
    from app.models import PostGenerationUsage

    insp = inspect(engine)
    if not insp.has_table("post_generations"):
        return
    pg_cols = {c["name"] for c in insp.get_columns("post_generations")}
    if not insp.has_table("post_generation_usage"):
        Base.metadata.create_all(bind=engine, tables=[PostGenerationUsage.__table__])
        insp = inspect(engine)

    if "usage_breakdown" not in pg_cols or "estimated_cost_usd" not in pg_cols:
        return

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO post_generation_usage (post_id, usage_breakdown, estimated_cost_usd)
                SELECT
                    id,
                    IFNULL(usage_breakdown, '{}'),
                    IFNULL(estimated_cost_usd, 0)
                FROM post_generations
                WHERE NOT EXISTS (
                    SELECT 1 FROM post_generation_usage u
                    WHERE u.post_id = post_generations.id
                )
                """
            )
        )
    # SQLite 3.35+ supports DROP COLUMN
    for col in ("usage_breakdown", "estimated_cost_usd"):
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(f"ALTER TABLE post_generations DROP COLUMN {col}")
                )
        except Exception:
            pass


def ensure_post_generations_schema() -> None:
    """Create tables if missing; add newer columns; migrate legacy usage_* off post rows."""
    from app.models import PostGeneration, PostGenerationUsage

    engine = get_engine()
    insp = inspect(engine)
    if not insp.has_table("post_generations"):
        Base.metadata.create_all(
            bind=engine, tables=[PostGeneration.__table__, PostGenerationUsage.__table__]
        )
        return

    if not insp.has_table("post_generation_usage"):
        Base.metadata.create_all(bind=engine, tables=[PostGenerationUsage.__table__])

    _extra_columns: list[tuple[str, str]] = [
        (
            "engagement_package",
            "ALTER TABLE post_generations ADD COLUMN engagement_package TEXT DEFAULT '{}'",
        ),
        (
            "video_url",
            "ALTER TABLE post_generations ADD COLUMN video_url TEXT DEFAULT ''",
        ),
        (
            "media_type",
            "ALTER TABLE post_generations ADD COLUMN media_type TEXT DEFAULT 'image'",
        ),
        (
            "media_attribution",
            "ALTER TABLE post_generations ADD COLUMN media_attribution TEXT DEFAULT ''",
        ),
        (
            "session_summary",
            "ALTER TABLE post_generations ADD COLUMN session_summary TEXT DEFAULT ''",
        ),
    ]
    for col_name, alter_sql in _extra_columns:
        cols = {c["name"] for c in inspect(engine).get_columns("post_generations")}
        if col_name in cols:
            continue
        with engine.begin() as conn:
            conn.execute(text(alter_sql))

    _migrate_usage_off_post_row(engine)


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False},
        )
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
