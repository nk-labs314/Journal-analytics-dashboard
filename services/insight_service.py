import pandas as pd
import numpy as np
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from services import data_service

logger = logging.getLogger(__name__)

vader_analyzer = SentimentIntensityAnalyzer()


# ---------------------------
# Sentiment
# ---------------------------

def analyze_sentiment(text):
    try:
        return vader_analyzer.polarity_scores(text)['compound']
    except Exception:
        logger.exception("Sentiment analysis failed for text")
        return 0


# ---------------------------
# Correlation Interpretation
# ---------------------------

def interpret_correlation(corr):
    magnitude = abs(corr)

    if magnitude >= 0.6:
        strength = "Strong"
    elif magnitude >= 0.3:
        strength = "Moderate"
    elif magnitude > 0:
        strength = "Weak"
    else:
        return "No significant relationship"

    direction = "positive" if corr > 0 else "negative"
    return f"{strength} {direction} relationship"


# ---------------------------
# Mood Trend (slope-based)
# ---------------------------

def detect_mood_trend(user_id):
    try:
        df = data_service.get_recent_mood(user_id, limit=100)
    except Exception:
        logger.exception("Failed to fetch mood data for trend detection")
        return "Data unavailable"

    if len(df) < 7:
        return "Insufficient data (need 7+ entries)"

    df = df.sort_values('date')
    df['rolling_avg'] = df['mood_score'].rolling(window=7).mean()

    recent_values = df['rolling_avg'].dropna().tail(5)

    if len(recent_values) < 2:
        return "Insufficient data"

    x = np.arange(len(recent_values))
    y = recent_values.values

    slope, _ = np.polyfit(x, y, 1)

    if slope > 0.1:
        return "Improving"
    elif slope < -0.1:
        return "Declining"
    else:
        return "Stable"


# ---------------------------
# Behavior Alerts (No direct DB)
# ---------------------------

def analyze_behavior(user_id):
    try:
        df = data_service.get_recent_behavior(user_id, limit=100)
    except Exception:
        logger.exception("Failed to fetch behavior data")
        return []

    alerts = []

    if not df.empty:
        if df['sleep_hours'].mean() < 6:
            alerts.append("Low sleep detected")

        if df['social_interactions'].mean() < 2:
            alerts.append("Social withdrawal detected")

    return alerts