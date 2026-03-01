import pandas as pd
import numpy as np


def compute_dashboard_analysis(
    df_mood,
    df_behavior,
    df_all_journals,
    df_journals,
    interpret_correlation,
    analyze_sentiment,
    detect_mood_trend,
    analyze_behavior,
    user_id
):

    # ---------------------------
    # Aggregate to daily level
    # ---------------------------
    df_mood_daily = df_mood.groupby('date', as_index=False)['mood_score'].mean()
    df_behavior_daily = df_behavior.groupby('date', as_index=False)['sleep_hours'].mean()

    df_combined = pd.merge(
        df_mood_daily,
        df_behavior_daily,
        on='date',
        how='inner'
    )

    if len(df_combined) > 0:
        sleep_values = df_combined['sleep_hours'].tolist()
        mood_values_scatter = df_combined['mood_score'].tolist()
    else:
        sleep_values = []
        mood_values_scatter = []

    # ---------------------------
    # Basic metrics
    # ---------------------------
    avg_mood_7d = df_mood['mood_score'].mean() if not df_mood.empty else 0
    mood_volatility = df_mood['mood_score'].std() if not df_mood.empty else 0

    if mood_volatility < 1:
        volatility_label = "Stable"
    elif mood_volatility < 2:
        volatility_label = "Moderate variability"
    else:
        volatility_label = "High variability"

    avg_sleep = df_behavior['sleep_hours'].mean() if not df_behavior.empty else 0

    # ---------------------------
    # Correlation
    # ---------------------------
    if (
        len(df_combined) > 1
        and 'mood_score' in df_combined.columns
        and 'sleep_hours' in df_combined.columns
    ):
        correlation = df_combined[['mood_score', 'sleep_hours']].corr().iloc[0, 1]
    else:
        correlation = 0

    correlation_label = interpret_correlation(correlation)

    # ---------------------------
    # Sentiment
    # ---------------------------
    if not df_all_journals.empty:
        sentiments = df_all_journals['journal_entry'].apply(
            lambda x: analyze_sentiment(x)
        )
        avg_sentiment = sentiments.mean()
    else:
        avg_sentiment = 0

    # ---------------------------
    # Mood trend & alerts
    # ---------------------------
    mood_trend = detect_mood_trend(user_id)
    alerts = analyze_behavior(user_id)

    # ---------------------------
    # Rolling averages
    # ---------------------------
    df_mood['date'] = pd.to_datetime(df_mood['date'], errors='coerce')
    df_mood = df_mood.sort_values('date')
    df_mood['rolling_7'] = df_mood['mood_score'].rolling(window=7).mean()

    dates = df_mood['date'].dt.strftime('%Y-%m-%d').tolist()
    mood_values = df_mood['mood_score'].tolist()
    rolling_values = df_mood['rolling_7'].tolist()

    # ---------------------------
    # Stability Index
    # ---------------------------
    df_mood_daily['rolling_std'] = df_mood_daily['mood_score'].rolling(7).std()

    min_std = df_mood_daily['rolling_std'].min()
    max_std = df_mood_daily['rolling_std'].max()

    if pd.notna(min_std) and pd.notna(max_std) and max_std != min_std:
        df_mood_daily['normalized_volatility'] = (
            (df_mood_daily['rolling_std'] - min_std) / (max_std - min_std)
        )
    else:
        df_mood_daily['normalized_volatility'] = 0

    df_mood_daily['stability_index'] = 1 - df_mood_daily['normalized_volatility']
    stability_values = df_mood_daily['stability_index'].tolist()

    if not df_mood_daily.empty:
        current_stability = round(df_mood_daily['stability_index'].iloc[-1], 3)
        avg_stability = round(df_mood_daily['stability_index'].mean(), 3)
    else:
        current_stability = None
        avg_stability = None

    # ---------------------------
    # Trend Classification
    # ---------------------------
    trend_label = "Insufficient Data"

    recent_values = df_mood['rolling_7'].dropna().tail(5)

    if len(recent_values) >= 2:
        x = np.arange(len(recent_values))
        y = recent_values.values

        slope, _ = np.polyfit(x, y, 1)

        if slope > 0.1:
            trend_label = "Improving"
        elif slope < -0.1:
            trend_label = "Declining"
        else:
            trend_label = "Stable"

    # ---------------------------
    # Summary Generator
    # ---------------------------
    def generate_summary(trend_label, current_stability, volatility_label, correlation_label):

        trend_map = {
            "Improving": "Your mood has been improving recently.",
            "Declining": "Your mood has been declining recently.",
            "Stable": "Your mood has been relatively stable.",
            "Insufficient Data": "There is not enough data to determine a clear mood trend."
        }

        trend_text = trend_map.get(
            trend_label,
            "Mood trend could not be determined."
        )

        if current_stability is None:
            stability_text = "Stability could not be determined."
        else:
            stability_text = (
                "Emotional stability is high." if current_stability > 0.7
                else "Emotional stability is moderate." if current_stability > 0.4
                else "Emotional stability is low."
            )

        correlation_text = f"Sleep and mood relationship is {correlation_label.lower()}."

        return f"{trend_text} {stability_text} {correlation_text}"

    summary_text = generate_summary(
        trend_label,
        current_stability,
        volatility_label,
        correlation_label
    )

    # ---------------------------
    # Final Output
    # ---------------------------
    analysis = {
        'sentiment': avg_sentiment,
        'mood_trend': mood_trend,
        'alerts': alerts,
        'journals': df_journals.to_dict(orient='records'),
        'avg_mood_7d': round(avg_mood_7d, 2),
        'volatility_label': volatility_label,
        'mood_volatility': round(mood_volatility, 2),
        'avg_sleep': round(avg_sleep, 2),
        'correlation': round(correlation, 2),
        'correlation_label': correlation_label,
        'mood_dates': dates,
        'mood_values': mood_values,
        'rolling_values': rolling_values,
        'sleep_values': sleep_values,
        'mood_values_scatter': mood_values_scatter,
        'stability_values': stability_values,
        'current_stability': current_stability,
        'avg_stability': avg_stability,
        'summary_text': summary_text,
        'trend_label': trend_label
    }

    return analysis