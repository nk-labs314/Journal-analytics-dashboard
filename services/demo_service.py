import random
import logging
from datetime import date, timedelta
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from services.data_service import get_engine

logger = logging.getLogger(__name__)

DEMO_USER_ID = 3
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LOW_ENTRIES = [
    "Felt really drained today. Couldn't focus on anything and just wanted to stay in bed.",
    "Had a rough argument with a friend. Everything feels off right now.",
    "Anxiety was through the roof today. Even small tasks felt overwhelming.",
    "Couldn't sleep last night and it showed. My mood was terrible all day.",
    "Feeling really disconnected from everyone. Not sure what's going on with me.",
    "Did nothing productive today. Just sat with a heavy feeling in my chest.",
    "Everything irritated me today. I snapped at people I care about.",
    "Really struggling to see the point in things lately. Feeling quite hopeless.",
]

MID_ENTRIES = [
    "Average day. Got some work done, nothing exciting but nothing terrible either.",
    "Went for a short walk. Helped a little. Mood is okay.",
    "Had a decent conversation with a colleague. Felt a bit more like myself.",
    "Managed to finish a few tasks I'd been avoiding. Small win.",
    "Watched a show I like. Felt normal for a bit which was nice.",
    "Had lunch with a friend. Lifted my mood slightly.",
    "Did some reading. Felt calm and okay. Not great, not bad.",
    "Went to the gym. Body feels tired but mentally a bit clearer.",
    "Cooked a proper meal for once. Small thing but it mattered.",
    "Worked on a side project for a bit. Got into flow state for maybe an hour.",
]

HIGH_ENTRIES = [
    "Really good day. Everything clicked and I felt motivated and sharp.",
    "Had a great conversation with someone I haven't spoken to in a while. Felt energised.",
    "Woke up feeling rested and genuinely happy for no specific reason. Rode that wave all day.",
    "Crushed my to-do list and still had energy left over. Felt on top of everything.",
    "Went for a long run. The endorphins lasted all day. Feeling strong.",
    "Got some really positive feedback on work I've been doing. Confidence is high.",
    "Finished something I've been working on for weeks. Huge sense of relief and pride.",
]


def reset_demo_account():
    """Wipe and reseed all data for the demo account. Called on every demo login."""
    logger.info("Starting demo account reset for user_id=%s", DEMO_USER_ID)

    today = date.today()
    all_days = [today - timedelta(days=i) for i in range(1, 91)]
    selected_days = sorted(random.sample(all_days, 60))

    model = SentenceTransformer(MODEL_NAME)
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM EntryEmbeddings WHERE user_id = :uid"), {"uid": DEMO_USER_ID})
        conn.execute(text("DELETE FROM BehaviorData WHERE user_id = :uid"), {"uid": DEMO_USER_ID})
        conn.execute(text("DELETE FROM MoodLogs WHERE user_id = :uid"), {"uid": DEMO_USER_ID})

    for entry_date in selected_days:
        mood = random.choices(
            population=list(range(1, 11)),
            weights=[2, 3, 5, 8, 12, 15, 18, 15, 12, 10],
            k=1
        )[0]

        if mood <= 4:
            journal = random.choice(LOW_ENTRIES)
            sleep = round(random.uniform(3.5, 6.0), 1)
            activity = random.randint(1, 2)
            social = random.randint(0, 2)
        elif mood <= 7:
            journal = random.choice(MID_ENTRIES)
            sleep = round(random.uniform(6.0, 7.5), 1)
            activity = random.randint(2, 4)
            social = random.randint(1, 4)
        else:
            journal = random.choice(HIGH_ENTRIES)
            sleep = round(random.uniform(7.0, 9.0), 1)
            activity = random.randint(3, 5)
            social = random.randint(3, 5)

        with engine.begin() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO MoodLogs (user_id, mood_score, date, journal_entry)
                    VALUES (:user_id, :mood, :date, :journal)
                    RETURNING log_id
                """),
                {
                    "user_id": DEMO_USER_ID,
                    "mood": mood,
                    "date": entry_date.isoformat(),
                    "journal": journal,
                }
            )
            log_id = result.fetchone()[0]

            conn.execute(
                text("""
                    INSERT INTO BehaviorData (user_id, sleep_hours, activity_level, social_interactions, date)
                    VALUES (:user_id, :sleep, :activity, :social, :date)
                """),
                {
                    "user_id": DEMO_USER_ID,
                    "sleep": sleep,
                    "activity": activity,
                    "social": social,
                    "date": entry_date.isoformat(),
                }
            )

            
            embedding_bytes = None

            conn.execute(
                text("""
                    INSERT INTO EntryEmbeddings (user_id, log_id, embedding, date)
                    VALUES (:user_id, :log_id, :embedding, :date)
                """),
                {
                    "user_id": DEMO_USER_ID,
                    "log_id": log_id,
                    "embedding": embedding_bytes,
                    "date": entry_date.isoformat(),
                }
            )

    logger.info("Demo account reset complete — 60 entries inserted.")