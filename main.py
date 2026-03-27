import sys
import os
from functools import wraps
from datetime import date
from langdetect import detect
from flask import Flask, flash, render_template, request, redirect, session, url_for, jsonify, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import secrets
from services import analytics_service
from services import data_service
from services import insight_service
from config import Config
from sqlalchemy import text
from services.data_service import get_engine
import logging
from services.forecast_service import ForecastService
from services.lexicon_service import LexiconService
from services import auth_service
from services.embedding_service import EmbeddingService
from services.rag_service import RAGService
import threading


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_db():
    engine = get_engine()
    auth_users_sql = """
        CREATE TABLE IF NOT EXISTS AuthUsers (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """

    if engine.dialect.name != "sqlite":
        auth_users_sql = """
            CREATE TABLE IF NOT EXISTS AuthUsers (
                user_id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """

    entry_embeddings_sql = """
        CREATE TABLE IF NOT EXISTS EntryEmbeddings (
            id SERIAL PRIMARY KEY,
            user_id INTEGER NOT NULL,
            log_id INTEGER NOT NULL,
            embedding BYTEA NOT NULL,
            date TEXT NOT NULL
        )
    """

    if engine.dialect.name == "sqlite":
        entry_embeddings_sql = """
            CREATE TABLE IF NOT EXISTS EntryEmbeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                log_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                date TEXT NOT NULL
            )
        """

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS MoodUsers (
                user_id SERIAL PRIMARY KEY,
                age INTEGER,
                gender TEXT,
                baseline_mood INTEGER
            )
        """))

        conn.execute(text(auth_users_sql))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS MoodLogs (
                log_id SERIAL PRIMARY KEY,
                user_id INTEGER,
                mood_score INTEGER,
                date TEXT,
                journal_entry TEXT
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS BehaviorData (
                user_id INTEGER,
                sleep_hours REAL,
                activity_level INTEGER,
                social_interactions INTEGER,
                date TEXT
            )
        """))

        conn.execute(text(entry_embeddings_sql))


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.")
            return redirect(url_for("main.login"))
        return view_func(*args, **kwargs)

    return wrapper

from flask import Blueprint

main_bp = Blueprint("main", __name__)

forecast_service = None
lexicon_service = None
embedding_service = None
rag_service = None

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

@main_bp.before_request
def auth_and_csrf_checks():
    # Slide session expiry
    session.modified = True

    # CSRF Protection
    if request.method == "POST":
        token = session.get("_csrf_token")
        if not token or (token != request.form.get("csrf_token") and token != request.headers.get("X-CSRFToken")):
            abort(403)

    # Global active session invalidation (auth_hash matching)
    if "user_id" in session:
        try:
            auth_hash = auth_service.get_user_auth(session["user_id"])
        except Exception as e:
            logger.exception("Auth session check failed")
            session.clear()
            return redirect(url_for('main.login'))

        if not auth_hash or auth_hash != session.get("auth_hash"):
            session.clear()
            flash("Your session is invalid or your password was changed. Please log in again.")
            return redirect(url_for('main.login'))


def generate_csrf_token():
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_hex(32)
    return session["_csrf_token"]


@main_bp.route('/')
def home():
    return render_template("test.html")


@main_bp.route("/register", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def register():
    if "user_id" in session:
        return redirect(url_for("main.home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user_id, auth_hash, error = auth_service.register_user(username, password)
        if error:
            flash(error)
            return redirect(url_for("main.register"))

        session["user_id"] = user_id
        session["username"] = username
        session["auth_hash"] = auth_hash
        session.permanent = True
        flash("Account created.")
        return redirect(url_for("main.home"))

    return render_template("login.html", mode="register")


@main_bp.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def login():
    if "user_id" in session:
        return redirect(url_for("main.home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user_id, auth_hash, error = auth_service.login_user(username, password)
        if error:
            flash(error)
            return redirect(url_for("main.login"))

        session["user_id"] = user_id
        session["username"] = username
        session["auth_hash"] = auth_hash
        session.permanent = True
        flash("Logged in successfully.")
        return redirect(url_for("main.home"))

    return render_template("login.html", mode="login")


@main_bp.route("/logout")
def logout():
    session.clear()
    flash("Logged out.")
    return redirect(url_for("main.login"))

@main_bp.route('/log', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def log_entry():
    user_id = session["user_id"]
    data = {
        'mood': int(request.form['mood']),
        'journal': request.form['journal'],
        'sleep': float(request.form['sleep']),
        'activity': int(request.form['activity']),
        'social': int(request.form['social'])
    }
    data_service.insert_mood_log(
        user_id,
        data['mood'],
        data['journal']
    )

    data_service.insert_behavior_log(
        user_id,
        data['sleep'],
        data['activity'],
        data['social']
    )

    # Generate and store embedding for the journal entry
    try:
        embedding = embedding_service.embed(data['journal'])
        log_id = data_service.get_last_log_id(user_id)
        if log_id:
            embedding_bytes = embedding.astype('float32').tobytes()
            data_service.insert_embedding(user_id, log_id, embedding_bytes, date.today().isoformat())
            logger.info("Embedding stored for user %s, log %s", user_id, log_id)
    except Exception:
        logger.exception("Failed to generate/store embedding for user %s", user_id)
        # Don't fail the whole request if embedding fails

    flash("Entry saved successfully!")
    return redirect(url_for('main.home'))



@main_bp.route('/dashboard')
@login_required
def dashboard():
    user_id = session["user_id"]

    df_mood = data_service.get_recent_mood(user_id)
    df_behavior = data_service.get_recent_behavior(user_id)
    df_all_journals = data_service.get_all_journals(user_id)

    analysis = analytics_service.compute_dashboard_analysis(
        df_mood,
        df_behavior,
        df_all_journals,
        df_all_journals,
        insight_service.interpret_correlation,
        insight_service.analyze_sentiment,
        insight_service.detect_mood_trend,
        insight_service.analyze_behavior,
        user_id
    )
    return render_template('dashboard.html', analysis=analysis)


@main_bp.route('/journals')
@login_required
def journals():
    user_id = session["user_id"]
    df_journals = data_service.get_all_journals(user_id)

    return render_template('journals.html', journals=df_journals.to_dict(orient='records'))

@main_bp.route("/forecast")
@login_required
def forecast():
    user_id = session["user_id"]

    user_df = data_service.get_all_journals(user_id)
    predictions = forecast_service.predict(user_df)

    return render_template(
        "forecast.html",
        predictions=predictions
    )

@main_bp.route("/insights", methods=["GET", "POST"])
@login_required
@limiter.limit("10 per minute")
def insights():
    result = None
    contributions = []

    if request.method == "POST":
        text = request.form["text"]
        user_id = session["user_id"]

        # Pass user's journal history so personalisation kicks in
        user_df = data_service.get_all_journals(user_id)

        prediction, contributions = lexicon_service.analyze_text(text, user_df, user_id)
        result = round(prediction, 2)

    return render_template(
        "insights.html",
        result=result,
        contributions=contributions
    )


@main_bp.route("/chat")
@login_required
def chat():
    return render_template("chat.html")


@main_bp.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        action = request.form.get("action")

        if action == "change_password":
            old_pw = request.form.get("old_password", "")
            new_pw = request.form.get("new_password", "")
            if not old_pw or not new_pw:
                flash("Both fields are required.")
                return redirect(url_for("main.settings"))

            success, result = auth_service.change_password(session["user_id"], old_pw, new_pw)
            if success:
                session["auth_hash"] = result  # new auth_hash
                flash("Password changed successfully.")
            else:
                flash(result)
            return redirect(url_for("main.settings"))

        elif action == "delete_account":
            password = request.form.get("password", "")
            success, error = auth_service.delete_account(session["user_id"], password)
            if success:
                session.clear()
                flash("Account deleted.")
                return redirect(url_for("main.login"))
            else:
                flash(error)
                return redirect(url_for("main.settings"))

    return render_template("settings.html")


# Keywords that indicate a mood/journal-related query
MOOD_KEYWORDS = {
    "mood", "feel", "feeling", "felt", "sleep", "journal", "sad", "happy",
    "anxiety", "anxious", "stress", "stressed", "pattern", "trend", "forecast",
    "depressed", "depression", "emotion", "emotional", "angry", "anger",
    "tired", "energy", "activity", "social", "well-being", "wellbeing",
    "mental", "health", "score", "entry", "entries", "week", "month",
    "improve", "decline", "better", "worse", "why", "how", "what",
}


@main_bp.route("/chat/message", methods=["POST"])
@login_required
@limiter.limit("5 per minute")
def chat_message():
    data = request.get_json(silent=True)
    if not data or not data.get("query", "").strip():
        return jsonify({"error": "Query is required."}), 400

    query = data["query"].strip()
    history = data.get("history", [])
    user_id = session["user_id"]

    # Retrieve similar entries
    retrieved = rag_service.retrieve(query, user_id)

    # Relevance pre-check: if no entries found and query has no mood keywords, refuse
    query_lower = query.lower()
    query_words = set(query_lower.split())
    if not retrieved and not query_words.intersection(MOOD_KEYWORDS):
        return jsonify({
            "response": "I can only answer questions about your journal and mood data."
        })

    # Get analytics context
    try:
        df_mood = data_service.get_recent_mood(user_id)
        df_behavior = data_service.get_recent_behavior(user_id)
        df_all_journals = data_service.get_all_journals(user_id)

        analytics = analytics_service.compute_dashboard_analysis(
            df_mood, df_behavior, df_all_journals, df_all_journals,
            insight_service.interpret_correlation,
            insight_service.analyze_sentiment,
            insight_service.detect_mood_trend,
            insight_service.analyze_behavior,
            user_id
        )
    except Exception:
        logger.exception("Failed to compute analytics for chat context")
        analytics = {}

    # Get forecast
    try:
        user_df = data_service.get_all_journals(user_id)
        predictions = forecast_service.predict(user_df) or {}
    except Exception:
        logger.exception("Failed to get forecast for chat context")
        predictions = {}

    response = rag_service.generate(query, retrieved, analytics, predictions, history=history)

    return jsonify({"response": response})


@main_bp.route("/health")
def health():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "new_code", "database": "ok"}, 200
    except Exception:
        logger.exception("Health check failed")
        return {"status": "error", "database": "error"}, 500


@main_bp.errorhandler(500)
def handle_500(error):
    logger.exception("Internal server error occurred")
    return {"error": "Internal server error"}, 500


@main_bp.errorhandler(404)
def handle_404(error):
    return {"error": "Resource not found"}, 404

def warm_embedding_model():
    try:
        logger.info("Background embedding warmup starting...")
        embedding_service.embed("warmup")
        logger.info("Background embedding warmup finished.")
    except Exception:
        logger.exception("Background embedding warmup failed")

def create_app(config_class=Config):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    app_instance = Flask(
        __name__,
        template_folder=os.path.join(base_dir, "templates"),
        static_folder=os.path.join(base_dir, "static"),
    )
    app_instance.config.from_object(config_class)
    app_instance.config.update(
        SESSION_COOKIE_SAMESITE="None",
        SESSION_COOKIE_SECURE=True
    )
    app_instance.secret_key = app_instance.config["SECRET_KEY"]

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        level=logging.INFO,
        force=True
    )

    app_instance.jinja_env.globals['csrf_token'] = generate_csrf_token

    limiter.init_app(app_instance)

    global forecast_service, lexicon_service, embedding_service, rag_service
    forecast_service = ForecastService()
    lexicon_service = LexiconService()
    embedding_service = EmbeddingService()
    rag_service = RAGService(embedding_service)

    if app_instance.config.get("TESTING"):
        pass

    # CORS headers
    @app_instance.after_request
    def add_cors_headers(response):
        allowed = app_instance.config.get("CORS_ORIGINS", "*")
        response.headers["Access-Control-Allow-Origin"] = allowed
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-CSRFToken"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

    app_instance.register_blueprint(main_bp)

    with app_instance.app_context():
        try:
            init_db()
        except Exception:
            logger.exception("Database initialization failed")
    threading.Thread(target=warm_embedding_model, daemon=True).start()
    return app_instance

app = create_app()

if __name__ == '__main__':
    app.run(debug=app.config["DEBUG"])
