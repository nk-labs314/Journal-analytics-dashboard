import sqlite3
import pandas as pd
import psycopg2
from urllib.parse import urlparse
from datetime import date
import os
from config import Config

def is_sqlite():
    return Config.DATABASE_URL.startswith("sqlite:///")


def get_placeholder():
    return "?" if is_sqlite() else "%s"

def get_connection():
    db_url = Config.DATABASE_URL

    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        return sqlite3.connect(db_path)

    if db_url.startswith("postgres://") or db_url.startswith("postgresql://"):
        result = urlparse(db_url)

        return psycopg2.connect(
            database=result.path[1:],
            user=result.username,
            password=result.password,
            host=result.hostname,
            port=result.port
        )

    raise ValueError("Unsupported DATABASE_URL format")


# --------------------------
# INSERT OPERATIONS
# --------------------------

def insert_mood_log(user_id, mood, journal):
    conn = get_connection()
    cursor = conn.cursor()

    placeholder = get_placeholder()

    query = f'''
    INSERT INTO MoodLogs (user_id, mood_score, date, journal_entry)
    VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder})
    '''

    cursor.execute(
        query,
        (user_id, mood, date.today().isoformat(), journal)
    )

    conn.commit()
    conn.close()


def insert_behavior_log(user_id, sleep, activity, social):
    conn = get_connection()
    cursor = conn.cursor()
    placeholder=get_placeholder()
    query = f'''
        INSERT INTO BehaviorData (user_id, sleep_hours, activity_level, social_interactions, date)
        VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
        '''
    cursor.execute(
        query,
        (user_id, sleep, activity, social, date.today().isoformat())
    )
    conn.commit()
    conn.close()


# --------------------------
# SELECT OPERATIONS
# --------------------------

def get_recent_mood(user_id, limit=30):
    conn = get_connection()

    placeholder = get_placeholder()

    query = f'''
    SELECT *
    FROM MoodLogs
    WHERE user_id={placeholder} AND date != {placeholder}
    ORDER BY date DESC
    LIMIT {placeholder}
    '''

    df = pd.read_sql(query, conn, params=(user_id, '2023-01-01', limit))

    conn.close()
    return df


def get_recent_behavior(user_id, limit=30):
    conn = get_connection()
    placeholder = get_placeholder()
    query = f'''
    SELECT *
    FROM BehaviorData
    WHERE user_id={placeholder} AND date != {placeholder}
    ORDER BY date DESC
    LIMIT {placeholder}
    '''
        
    df = pd.read_sql(query, conn, params=(user_id, '2023-01-01', limit))
    conn.close()
    return df


def get_all_journals(user_id):
    conn = get_connection()
    placeholder = get_placeholder()
    query = f'''
    SELECT date, journal_entry, mood_score
    FROM MoodLogs
    WHERE user_id={placeholder} AND date != {placeholder}
    ORDER BY date DESC
    '''
        
    df = pd.read_sql(query, conn, params=(user_id, '2023-01-01'))
    
    conn.close()
    return df