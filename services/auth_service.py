import logging
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash
from services.data_service import get_engine
from services.demo_service import reset_demo_account

logger = logging.getLogger(__name__)

DEMO_USERNAME = "demo_acc"


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
    except IntegrityError:
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

    # Return (user_id, shortened hash string for session invalidation)
    return int(row.user_id), row.password_hash[:10]


def get_user_auth(user_id):
    query = text("SELECT password_hash FROM AuthUsers WHERE user_id = :user_id")
    with get_engine().connect() as conn:
        row = conn.execute(query, {"user_id": user_id}).fetchone()
    return row.password_hash[:10] if row else None


def change_password(user_id, old_password, new_password):
    engine = get_engine()

    # Verify old
    query = text("SELECT password_hash FROM AuthUsers WHERE user_id = :user_id")
    with engine.connect() as conn:
        row = conn.execute(query, {"user_id": user_id}).fetchone()

    if not row or not check_password_hash(row.password_hash, old_password):
        return False, "Incorrect old password."

    new_hash = generate_password_hash(new_password)
    update_query = text("UPDATE AuthUsers SET password_hash = :new_hash WHERE user_id = :user_id")
    
    try:
        with engine.begin() as conn:
            conn.execute(update_query, {"new_hash": new_hash, "user_id": user_id})
        return True, new_hash[:10]
    except Exception:
        logger.exception("Failed to change password")
        return False, "Database error."


def delete_account(user_id, password):
    engine = get_engine()

    query = text("SELECT password_hash FROM AuthUsers WHERE user_id = :user_id")
    with engine.connect() as conn:
        row = conn.execute(query, {"user_id": user_id}).fetchone()

    if not row or not check_password_hash(row.password_hash, password):
        return False, "Incorrect password."

    # Drop all related data
    try:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM MoodLogs WHERE user_id = :user_id"), {"user_id": user_id})
            conn.execute(text("DELETE FROM BehaviorData WHERE user_id = :user_id"), {"user_id": user_id})
            conn.execute(text("DELETE FROM EntryEmbeddings WHERE user_id = :user_id"), {"user_id": user_id})
            conn.execute(text("DELETE FROM MoodUsers WHERE user_id = :user_id"), {"user_id": user_id})
            conn.execute(text("DELETE FROM AuthUsers WHERE user_id = :user_id"), {"user_id": user_id})
        return True, None
    except Exception:
        logger.exception("Failed to delete account")
        return False, "Database error."


def login_user(username, password):
    """Authenticate a user and handle demo account reset.

    Returns:
        (user_id, None) on success, (None, error_message) on failure.
    """
    if not username or not password:
        return None, "Username and password are required."

    try:
        result = verify_user(username, password)
    except Exception as e:
        logger.exception("Login failed due to database error")
        return None, None, "Login failed. Please try again in a moment."

    if result is None:
        return None, None, "Invalid username or password."
        
    user_id, auth_hash = result

    if username == DEMO_USERNAME:
        try:
            reset_demo_account()
        except Exception:
            logger.exception("Demo reset failed")

    return user_id, auth_hash, None


def register_user(username, password):
    """Register a new user account.

    Returns:
        (user_id, None) on success, (None, error_message) on failure.
    """
    if not username or not password:
        return None, "Username and password are required."

    try:
        created = create_user(username, password)
    except Exception:
        logger.exception("Registration failed due to database error")
        return None, None, "Registration failed. Please try again in a moment."

    if not created:
        return None, None, "Username already exists."

    return verify_user(username, password) + (None,)
