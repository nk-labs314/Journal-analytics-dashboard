import os
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()

class Config:
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY")
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY must be set")

    PERMANENT_SESSION_LIFETIME = timedelta(minutes=30)

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///mental_health.db")

    # Debug
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # External APIs (SAFE to default empty)
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

    ENV = os.getenv("ENV", "development")

    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SECURE = False

    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")


class TestingConfig(Config):
    TESTING = True
    DATABASE_URL = "sqlite:///:memory:"
    RATELIMIT_ENABLED = False