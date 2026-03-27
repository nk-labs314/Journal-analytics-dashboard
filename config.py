import os
from dotenv import load_dotenv
from datetime import timedelta

# Load secret variables from .env file into standard environment variables
load_dotenv()

class Config:
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key")
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=30)

    # Database
    # Default: SQLite file in project root
    # For Supabase: set DATABASE_URL to your Supabase PostgreSQL connection string
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///mental_health.db")

    # Debug mode
    DEBUG = os.getenv("DEBUG", "True") == "True"

    # Hugging Face API (for RAG)
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
   


class TestingConfig(Config):
    TESTING = True
    DATABASE_URL = "sqlite:///:memory:"
    RATELIMIT_ENABLED = False
    