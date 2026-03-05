from sqlalchemy import text
from werkzeug.security import check_password_hash, generate_password_hash
from services.data_service import get_engine


def create_user(username, password):
    engine = get_engine()
    password_hash = generate_password_hash(password)

    query = text("""
        INSERT INTO AuthUsers (username, password_hash)
        VALUES (:username, :password_hash)
    """)

    try:
        with engine.begin() as conn:
            conn.execute(
                query,
                {
                    "username": username,
                    "password_hash": password_hash,
                },
            )
        return True
    except Exception:
        return False


def verify_user(username, password):
    engine = get_engine()

    query = text("""
        SELECT user_id, password_hash
        FROM AuthUsers
        WHERE username = :username
    """)

    with engine.connect() as conn:
        row = conn.execute(query, {"username": username}).fetchone()

    if row is None:
        return None

    if not check_password_hash(row.password_hash, password):
        return None

    return int(row.user_id)
