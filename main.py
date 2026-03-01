import sys
import os
from langdetect import detect
from flask import Flask, flash, render_template, request, redirect, url_for
from services import analytics_service
from services import data_service
from services import insight_service
from config import Config
from services.data_service import get_connection



Default_User_Id = 1



def init_db():
    conn = get_connection()
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

app = Flask(__name__)
app.static_folder = 'static'
app.config.from_object(Config)
app.secret_key = app.config["SECRET_KEY"]

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
    insight_service.interpret_correlation,
    insight_service.analyze_sentiment,
    insight_service.detect_mood_trend,
    insight_service.analyze_behavior,
    user_id
    )    
    return render_template('dashboard.html', analysis=analysis)


@app.route('/journals')
def journals():
    user_id = Default_User_Id
    df_journals = data_service.get_all_journals(user_id)

    return render_template('journals.html', journals=df_journals.to_dict(orient='records'))


#sys.path.append(os.path.join(os.getcwd(), "Mental-health-Chatbot"))
#def detect_language(user_input):
   # try:
       # if len(user_input.split()) < 5:
       #     return 'en'
       # detected_lang = detect(user_input)
        #if detected_lang not in ['en']:
       #     return 'en'
       # return detected_lang
   # except Exception as e:
       # print(f"[ERROR] Language detection failed: {str(e)}")
        #return 'en'

#def chat_response(user_input):
 #   if not user_input.strip():
  #      return "Please enter a message."
   # try:
    #    response_text = specialized_chatbot_response(user_input)
     #   return response_text
    #except Exception as e:
      #  print(f"[ERROR] Chatbot generation failed: {str(e)}")
       # return "I'm having trouble processing that. Could you rephrase?"

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
    app.run(debug=Config.DEBUG)

