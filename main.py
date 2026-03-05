import sys
import os
from functools import wraps
from langdetect import detect
from flask import Flask, flash, render_template, request, redirect, session, url_for
from services import analytics_service
from services import data_service
from services import insight_service
from config import Config
from sqlalchemy import text
from services.data_service import get_engine
import logging
from services.forecast_service import ForecastService
from services.data_service import get_all_journals  
from services.lexicon_service import LexiconService
from services import auth_service

forecast_service = ForecastService()
lexicon_service = LexiconService()

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

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS Users (
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
    conn.close()


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapper

app = Flask(__name__)
app.static_folder = 'static'
app.config.from_object(Config)
app.secret_key = app.config["SECRET_KEY"]

@app.route('/')
@login_required
def home():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Username and password are required.")
            return redirect(url_for("register"))

        created = auth_service.create_user(username, password)
        if not created:
            flash("Username already exists.")
            return redirect(url_for("register"))

        user_id = auth_service.verify_user(username, password)
        session["user_id"] = user_id
        session["username"] = username
        flash("Account created.")
        return redirect(url_for("home"))

    return render_template("login.html", mode="register")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user_id = auth_service.verify_user(username, password)
        if user_id is None:
            flash("Invalid username or password.")
            return redirect(url_for("login"))

        session["user_id"] = user_id
        session["username"] = username
        flash("Logged in successfully.")
        return redirect(url_for("home"))

    return render_template("login.html", mode="login")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.")
    return redirect(url_for("login"))

@app.route('/log', methods=['POST'])
@login_required
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

    flash("Entry saved successfully!")
    return redirect(url_for('home'))



@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session["user_id"]

    df_mood = data_service.get_recent_mood(user_id)
    df_behavior = data_service.get_recent_behavior(user_id)
    df_all_journals = data_service.get_all_journals(user_id)
    df_journals = data_service.get_all_journals(user_id)
    

    analysis = analytics_service.compute_dashboard_analysis(
    df_mood,
    df_behavior,
    df_all_journals,
    df_journals,
    insight_service.interpret_correlation,
    insight_service.analyze_sentiment,
    insight_service.detect_mood_trend,
    insight_service.analyze_behavior,
    user_id
    )    
    return render_template('dashboard.html', analysis=analysis)


@app.route('/journals')
@login_required
def journals():
    user_id = session["user_id"]
    df_journals = data_service.get_all_journals(user_id)

    return render_template('journals.html', journals=df_journals.to_dict(orient='records'))

@app.route("/forecast")
@login_required
def forecast():
    user_id = session["user_id"]

    user_df = get_all_journals(user_id)  # must return DataFrame
    predictions = forecast_service.predict(user_df)
    

    return render_template(
        "forecast.html",
        predictions=predictions
    )

@app.route("/insights", methods=["GET", "POST"])
@login_required
def insights():
    result = None
    contributions = []

    if request.method == "POST":
        text = request.form["text"]
        user_id = Default_User_Id

        # Pass user's journal history so personalisation kicks in
        user_df = data_service.get_all_journals(user_id)

        prediction, contributions = lexicon_service.analyze_text(text, user_df)
        result = round(prediction, 2)

    return render_template(
        "insights.html",
        result=result,
        contributions=contributions
    )

@app.route("/health")
def health():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        return {"status": "db_error"}, 500

    try:
        model_status = "loaded" if forecast_service.model else "missing"
    except Exception:
        model_status = "error"

    try:
        lexicon_status = "loaded" if lexicon_service.global_lexicon else "missing"
    except Exception:
        lexicon_status = "error"

    return {
        "status": "ok",
        "database": db_status,
        "forecast_model": model_status,
        "lexicon_model": lexicon_status
    }, 200



if Config.DEBUG:
    init_db()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.errorhandler(500)
def handle_500(error):
    logger.exception("Internal server error occurred")
    return {"error": "Internal server error"}, 500


@app.errorhandler(404)
def handle_404(error):
    return {"error": "Resource not found"}, 404
if __name__ == '__main__':
    app.run(debug=Config.DEBUG)

