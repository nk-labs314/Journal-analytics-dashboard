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

    return int(row.user_id)


def login_user(username, password):
    """Authenticate a user and handle demo account reset.

    Returns:
        (user_id, None) on success, (None, error_message) on failure.
    """
    if not username or not password:
        return None, "Username and password are required."

    try:
        user_id = verify_user(username, password)
    except Exception as e:
        logger.exception("Login failed due to database error")
        return None, "Login failed. Please try again in a moment."

    if user_id is None:
        return None, "Invalid username or password."

    if username == DEMO_USERNAME:
        try:
            reset_demo_account()
        except Exception:
            logger.exception("Demo reset failed")

    return user_id, None


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
        return None, "Registration failed. Please try again in a moment."

    if not created:
        return None, "Username already exists."

    user_id = verify_user(username, password)
    return user_id, None
