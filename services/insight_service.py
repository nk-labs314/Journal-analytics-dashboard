import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from services import data_service

vader_analyzer = SentimentIntensityAnalyzer()


# ---------------------------
# Sentiment
# ---------------------------

def analyze_sentiment(text):
    return vader_analyzer.polarity_scores(text)['compound']


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
# Mood Trend (No direct DB)
# ---------------------------

def detect_mood_trend(user_id):

    df = data_service.get_recent_mood(user_id, limit=100)

    if len(df) < 7:
        return "Insufficient data (need 7+ entries)"

    df = df.sort_values('date')
    df['rolling_avg'] = df['mood_score'].rolling(window=7).mean()

    if df['rolling_avg'].iloc[-1] < df['mood_score'].mean() - 1:
        return "Downward"

    return "Stable"


# ---------------------------
# Behavior Alerts (No direct DB)
# ---------------------------

def analyze_behavior(user_id):

    df = data_service.get_recent_behavior(user_id, limit=100)

    alerts = []

    if not df.empty:
        if df['sleep_hours'].mean() < 6:
            alerts.append("Low sleep detected")

        if df['social_interactions'].mean() < 2:
            alerts.append("Social withdrawal detected")

    return alerts