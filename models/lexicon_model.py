# lexicon_engine.py

import os
import logging
import numpy as np
from collections import defaultdict


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# Initialize once
lemmatizer = WordNetLemmatizer()
stop_words = None

NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "cannot", "can't", "don't",
    "didn't", "doesn't", "isn't", "aren't", "wasn't", "weren't", "hasn't",
    "haven't", "hadn't", "won't", "wouldn't", "shouldn't", "couldn't",
    "mightn't", "mustn't"
}

HIGH_VALENCE_OVERRIDES = {
    "kill": -4.0, "suicide": -5.0, "suicidal": -5.0, "worthless": -4.0,
    "hopeless": -4.0, "anxious": -2.0, "depressed": -3.0, "sad": -2.0,
    "productive": 2.0, "great": 2.0, "happy": 2.0
}


def _prepare_nltk_data():
    # In production (Docker), NLTK data is pre-downloaded via the Dockerfile.
    # In local development, the user should run:
    # python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
    
    # We still need to ensure the wordnet corpus is loaded to avoid the stale LazyCorpusLoader state bug
    try:
        from nltk.corpus import wordnet as wn
        if hasattr(wn, '_LazyCorpusLoader__args') or not hasattr(wn, '_morphy'):
            wn.ensure_loaded()
    except Exception:
        pass


# Ensure NLTK data is ready before any tokenization
_prepare_nltk_data()


def _get_stopwords():
    global stop_words
    if stop_words is not None:
        return stop_words

    _prepare_nltk_data()
    try:
        stop_words = set(stopwords.words("english"))
        stop_words = stop_words - NEGATION_WORDS
    except LookupError:
        logger.warning("NLTK stopwords unavailable; continuing with empty stopword set.")
        stop_words = set()
    return stop_words


# -----------------------------
# NLP Tokenizer
# -----------------------------
def tokenize(text):
    active_stop_words = _get_stopwords()

    try:
        tokens = word_tokenize(text.lower())
    except LookupError:
        _prepare_nltk_data()
        try:
            tokens = word_tokenize(text.lower())
        except LookupError:
            logger.warning("NLTK punkt unavailable; using regex fallback tokenizer.")
            tokens = text.lower().split()

    cleaned = []

    for token in tokens:
        if token.isalpha() and token not in active_stop_words:
            lemma = lemmatizer.lemmatize(token)
            cleaned.append(lemma)

    return cleaned


# -----------------------------
# Build Global Lexicon
# -----------------------------
def build_global_lexicon(df, min_freq=5):
    global_mean = df["mood_score"].mean()

    word_moods = defaultdict(list)

    for _, row in df.iterrows():
        words = set(tokenize(row["text"]))
        for w in words:
            word_moods[w].append(row["mood_score"])

    global_lexicon = {}
    global_counts = {}

    for w, moods in word_moods.items():
        if len(moods) >= min_freq:
            global_counts[w] = len(moods)
            global_lexicon[w] = np.mean(moods) - global_mean  # centered

    return global_lexicon, global_counts, global_mean


# -----------------------------
# Build User Lexicon
# -----------------------------
def build_user_lexicon(df, user_id, min_freq=3):
    user_df = df[df["user_id"] == user_id]
    user_mean = user_df["mood_score"].mean()

    word_moods = defaultdict(list)

    for _, row in user_df.iterrows():
        words = set(tokenize(row["text"]))
        for w in words:
            word_moods[w].append(row["mood_score"])

    user_lexicon = {}
    user_counts = {}

    for w, moods in word_moods.items():
        if len(moods) >= min_freq:
            user_counts[w] = len(moods)
            user_lexicon[w] = np.mean(moods) - user_mean  # centered to user

    return user_lexicon, user_counts


# -----------------------------
# Hybrid Score
# -----------------------------
def get_word_score(
    word,
    global_lexicon,
    global_counts,
    user_lexicon,
    user_counts,
    global_mean,
    k=10
):
    if word in HIGH_VALENCE_OVERRIDES:
        return HIGH_VALENCE_OVERRIDES[word]

    if word not in global_lexicon:
        return None

    global_score = global_lexicon[word]

    if word not in user_lexicon:
        return global_score

    n_user = user_counts[word]
    lambda_w = n_user / (n_user + k)

    user_score = user_lexicon[word]

    return lambda_w * user_score + (1 - lambda_w) * global_score


# -----------------------------
# Predict Mood From Text
# -----------------------------
def predict_mood_from_text(
    text,
    global_lexicon,
    global_counts,
    user_lexicon,
    user_counts,
    global_mean,
    k=10
):
    tokens = tokenize(text)

    scores = []
    contributions = []
    scored_words = set()

    for i, w in enumerate(tokens):
        if w in scored_words:
            continue

        base_score = get_word_score(
            w,
            global_lexicon,
            global_counts,
            user_lexicon,
            user_counts,
            global_mean,
            k
        )

        if base_score is not None:
            # Check for negation in the 2 preceding tokens
            window_start = max(0, i - 2)
            preceding_tokens = tokens[window_start:i]
            is_negated = any(t in NEGATION_WORDS for t in preceding_tokens)

            final_score = -base_score if is_negated else base_score

            scores.append(final_score)
            contributions.append((w, final_score))
            scored_words.add(w)

    if not scores:
        return global_mean, []

    centered_prediction = np.mean(scores)

    final_prediction = global_mean + centered_prediction

    contributions_sorted = sorted(
        contributions,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return final_prediction, contributions_sorted
