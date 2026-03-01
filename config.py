import os


class Config:
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key")

    # Database
    # Default: SQLite file in project root
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///mental_health.db")

    # Debug mode
    DEBUG = os.getenv("DEBUG", "True") == "True"