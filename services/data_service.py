import sqlite3
import pandas as pd
from datetime import date
import os
from config import Config


def get_connection():
    db_url = Config.DATABASE_URL

    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        return sqlite3.connect(db_path)

    raise ValueError("Unsupported DATABASE_URL format")


# --------------------------
# INSERT OPERATIONS
# --------------------------

def insert_mood_log(user_id, mood, journal):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        INSERT INTO MoodLogs (user_id, mood_score, date, journal_entry)
        VALUES (?, ?, ?, ?)
        ''',
        (user_id, mood, date.today().isoformat(), journal)
    )
    conn.commit()
    conn.close()


def insert_behavior_log(user_id, sleep, activity, social):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        INSERT INTO BehaviorData (user_id, sleep_hours, activity_level, social_interactions, date)
        VALUES (?, ?, ?, ?, ?)
        ''',
        (user_id, sleep, activity, social, date.today().isoformat())
    )
    conn.commit()
    conn.close()


# --------------------------
# SELECT OPERATIONS
# --------------------------

def get_recent_mood(user_id, limit=30):
    conn = get_connection()
    df = pd.read_sql(
        '''
        SELECT *
        FROM MoodLogs
        WHERE user_id=? AND date != ?
        ORDER BY date DESC
        LIMIT ?
        ''',
        conn,
        params=(user_id, '2023-01-01', limit)
    )
    conn.close()
    return df


def get_recent_behavior(user_id, limit=30):
    conn = get_connection()
    df = pd.read_sql(
        '''
        SELECT *
        FROM BehaviorData
        WHERE user_id=? AND date != ?
        ORDER BY date DESC
        LIMIT ?
        ''',
        conn,
        params=(user_id, '2023-01-01', limit)
    )
    conn.close()
    return df


def get_all_journals(user_id):
    conn = get_connection()
    df = pd.read_sql(
        '''
        SELECT date, journal_entry, mood_score
        FROM MoodLogs
        WHERE user_id=? AND date != ?
        ORDER BY date DESC
        ''',
        conn,
        params=(user_id, '2023-01-01')
    )
    conn.close()
    return df