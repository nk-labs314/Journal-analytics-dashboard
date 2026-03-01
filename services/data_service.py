from sqlalchemy import create_engine, text
import pandas as pd
from datetime import date
from config import Config


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


def insert_behavior_log(user_id, sleep, activity, social):
    engine = get_engine()

    query = text("""
        INSERT INTO BehaviorData (user_id, sleep_hours, activity_level, social_interactions, date)
        VALUES (:user_id, :sleep, :activity, :social, :date)
    """)

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


# --------------------------
# SELECT OPERATIONS
# --------------------------

def get_recent_mood(user_id, limit=30):
    engine = get_engine()

    query = text("""
        SELECT *
        FROM MoodLogs
        WHERE user_id = :user_id AND date != :excluded_date
        ORDER BY date DESC
        LIMIT :limit
    """)

    return pd.read_sql(
        query,
        engine,
        params={
            "user_id": user_id,
            "excluded_date": "2023-01-01",
            "limit": limit
        }
    )


def get_recent_behavior(user_id, limit=30):
    engine = get_engine()

    query = text("""
        SELECT *
        FROM BehaviorData
        WHERE user_id = :user_id AND date != :excluded_date
        ORDER BY date DESC
        LIMIT :limit
    """)

    return pd.read_sql(
        query,
        engine,
        params={
            "user_id": user_id,
            "excluded_date": "2023-01-01",
            "limit": limit
        }
    )


def get_all_journals(user_id):
    engine = get_engine()

    query = text("""
        SELECT date, journal_entry, mood_score
        FROM MoodLogs
        WHERE user_id = :user_id AND date != :excluded_date
        ORDER BY date DESC
    """)

    return pd.read_sql(
        query,
        engine,
        params={
            "user_id": user_id,
            "excluded_date": "2023-01-01"
        }
    )