from statistics import correlation
import sys
import os
import sqlite3
import pandas as pd
import re
import json
import random
import numpy as np
import nltk
from datetime import date
from flask import Flask, flash, render_template, request, redirect, url_for
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from services import analytics_service
from services import data_service


sys.path.append(os.path.join(os.getcwd(), "Mental-health-Chatbot"))
Default_User_Id = 1
DB_PATH = os.path.join(os.getcwd(), 'mental_health.db')


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            user_id INTEGER PRIMARY KEY,
            age INTEGER,
            gender TEXT,
            baseline_mood INTEGER
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS MoodLogs (
            log_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            mood_score INTEGER,
            date TEXT,
            journal_entry TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS BehaviorData (
            user_id INTEGER,
            sleep_hours REAL,
            activity_level INTEGER,
            social_interactions INTEGER,
            date TEXT
        )
    ''')
   
    conn.commit()
    conn.close()



vader_analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    return vader_analyzer.polarity_scores(text)['compound']

def detect_mood_trend(user_id):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT mood_score, date
        FROM MoodLogs
        WHERE user_id = ?
        ORDER BY date
    ''', conn, params=(user_id,))
    conn.close()
    if len(df) < 7:
        return "Insufficient data (need 7+ entries)"
    df['rolling_avg'] = df['mood_score'].rolling(window=7).mean()
    return "Downward" if df['rolling_avg'].iloc[-1] < df['mood_score'].mean() - 1 else "Stable"

def analyze_behavior(user_id):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT sleep_hours, social_interactions
        FROM BehaviorData
        WHERE user_id = ?
    ''', conn, params=(user_id,))
    conn.close()
    alerts = []
    if not df.empty:
        if df['sleep_hours'].mean() < 6:
            alerts.append("Low sleep detected")
        if df['social_interactions'].mean() < 2:
            alerts.append("Social withdrawal detected")
    return alerts

from langdetect import detect

def detect_language(user_input):
    try:
        if len(user_input.split()) < 5:
            return 'en'
        detected_lang = detect(user_input)
        if detected_lang not in ['en']:
            return 'en'
        return detected_lang
    except Exception as e:
        print(f"[ERROR] Language detection failed: {str(e)}")
        return 'en'

#def chat_response(user_input):
 #   if not user_input.strip():
  #      return "Please enter a message."
   # try:
    #    response_text = specialized_chatbot_response(user_input)
     #   return response_text
    #except Exception as e:
      #  print(f"[ERROR] Chatbot generation failed: {str(e)}")
       # return "I'm having trouble processing that. Could you rephrase?"

app = Flask(__name__)
app.static_folder = 'static'
app.secret_key = "key"

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/log', methods=['POST'])
def log_entry():
    user_id = Default_User_Id
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

@app.route('/dashboard')
def dashboard():
    user_id = Default_User_Id

    df_mood = data_service.get_recent_mood(user_id)
    df_behavior = data_service.get_recent_behavior(user_id)
    df_all_journals = data_service.get_all_journals(user_id)
    df_journals = data_service.get_all_journals(user_id)
    

    analysis = analytics_service.compute_dashboard_analysis(
        df_mood,
        df_behavior,
        df_all_journals,
        df_journals,
        interpret_correlation,
        analyze_sentiment,
        detect_mood_trend,
        analyze_behavior,
        user_id
    )


    
    return render_template('dashboard.html', analysis=analysis)
@app.route('/journals')
def journals():
    user_id = Default_User_Id
    conn = sqlite3.connect(DB_PATH)
    df_journals = pd.read_sql(
        'SELECT date, journal_entry, mood_score FROM MoodLogs WHERE user_id=? ORDER BY date DESC',
        conn,
        params=(user_id,)
    )
    conn.close()

    return render_template('journals.html', journals=df_journals.to_dict(orient='records'))




#@app.route('/chat', methods=['GET', 'POST'])
#def chat():
 #   response = None
  #  user_input = ""
   # if request.method == 'POST':
    #    user_input = request.form.get('message', '').strip()
     #   if user_input:
      #      detected_lang = detect_language(user_input)
       #     print("detected english")
        #    response = chat_response(user_input)
        #return f'<div id="bot-response">{response}</div>'
    #return render_template('chat.html', response=response, user_input=user_input)
init_db()
if __name__ == '__main__':
    app.run(debug=True)

