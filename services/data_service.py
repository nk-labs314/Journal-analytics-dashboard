from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import logging
from datetime import date
from config import Config

logger = logging.getLogger(__name__)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(Config.DATABASE_URL, future=True)
    return _engine


# --------------------------
# INSERT OPERATIONS
# --------------------------

def insert_mood_log(user_id, mood, journal):
    engine = get_engine()

    query = text("""
        INSERT INTO MoodLogs (user_id, mood_score, date, journal_entry)
        VALUES (:user_id, :mood, :date, :journal)
    """)

    try:
        with engine.begin() as conn:
            conn.execute(
                query,
                {
                    "user_id": user_id,
                    "mood": mood,
                    "date": date.today().isoformat(),
                    "journal": journal
                }
            )
    except Exception:
        logger.exception("Failed to insert mood log for user %s", user_id)
        raise


def insert_behavior_log(user_id, sleep, activity, social):
    engine = get_engine()

    query = text("""
        INSERT INTO BehaviorData (user_id, sleep_hours, activity_level, social_interactions, date)
        VALUES (:user_id, :sleep, :activity, :social, :date)
    """)

    try:
        with engine.begin() as conn:
            conn.execute(
                query,
                {
                    "user_id": user_id,
                    "sleep": sleep,
                    "activity": activity,
                    "social": social,
                    "date": date.today().isoformat()
                }
            )
    except Exception:
        logger.exception("Failed to insert behavior log for user %s", user_id)
        raise


# --------------------------
# SELECT OPERATIONS
# --------------------------

def get_recent_mood(user_id, limit=30):
    engine = get_engine()

    query = text("""
        SELECT *
        FROM MoodLogs
        WHERE user_id = :user_id
        ORDER BY date DESC
        LIMIT :limit
    """)

    try:
        return pd.read_sql(
            query,
            engine,
            params={
                "user_id": user_id,
                "limit": limit
            }
        )
    except Exception:
        logger.exception("Failed to fetch recent mood for user %s", user_id)
        raise


def get_recent_behavior(user_id, limit=30):
    engine = get_engine()

    query = text("""
        SELECT *
        FROM BehaviorData
        WHERE user_id = :user_id
        ORDER BY date DESC
        LIMIT :limit
    """)

    try:
        return pd.read_sql(
            query,
            engine,
            params={
                "user_id": user_id,
                "limit": limit
            }
        )
    except Exception:
        logger.exception("Failed to fetch recent behavior for user %s", user_id)
        raise


def get_all_journals(user_id):
    engine = get_engine()

    query = text("""
        SELECT date, journal_entry, mood_score
        FROM MoodLogs
        WHERE user_id = :user_id
        ORDER BY date DESC
    """)

    try:
        return pd.read_sql(
            query,
            engine,
            params={
                "user_id": user_id
            }
        )
    except Exception:
        logger.exception("Failed to fetch journals for user %s", user_id)
        raise


# --------------------------
# EMBEDDING OPERATIONS
# --------------------------

def insert_embedding(user_id, log_id, embedding_bytes, entry_date):
    engine = get_engine()

    query = text("""
        INSERT INTO EntryEmbeddings (user_id, log_id, embedding, date)
        VALUES (:user_id, :log_id, :embedding, :date)
    """)

    try:
        with engine.begin() as conn:
            conn.execute(
                query,
                {
                    "user_id": user_id,
                    "log_id": log_id,
                    "embedding": embedding_bytes,
                    "date": entry_date
                }
            )
    except Exception:
        logger.exception("Failed to insert embedding for user %s, log %s", user_id, log_id)
        raise


def get_last_log_id(user_id):
    engine = get_engine()

    query = text("""
        SELECT log_id FROM MoodLogs
        WHERE user_id = :user_id
        ORDER BY log_id DESC
        LIMIT 1
    """)

    try:
        with engine.connect() as conn:
            row = conn.execute(query, {"user_id": user_id}).fetchone()
        return row.log_id if row else None
    except Exception:
        logger.exception("Failed to get last log_id for user %s", user_id)
        raise


def get_embeddings_for_user(user_id):
    engine = get_engine()

    query = text("""
        SELECT e.log_id, e.date, e.embedding, m.journal_entry, m.mood_score
        FROM EntryEmbeddings e
        JOIN MoodLogs m ON e.log_id = m.log_id
        WHERE e.user_id = :user_id
        ORDER BY e.date DESC
    """)

    try:
        with engine.connect() as conn:
            rows = conn.execute(query, {"user_id": user_id}).fetchall()

        return [
            {
                "log_id": row.log_id,
                "date": row.date,
                "embedding": row.embedding,
                "journal_entry": row.journal_entry,
                "mood_score": row.mood_score
            }
            for row in rows
        ]
    except Exception:
        logger.exception("Failed to fetch embeddings for user %s", user_id)
        raise
