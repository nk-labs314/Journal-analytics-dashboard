# lexicon_engine.py

import re
import numpy as np
import pandas as pd
from collections import defaultdict


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Initialize once
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# -----------------------------
# NLP Tokenizer
# -----------------------------
def tokenize(text):
    tokens = word_tokenize(text.lower())

    cleaned = []

    for token in tokens:
        if token.isalpha() and token not in stop_words:
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
    words = tokenize(text)

    scores = []
    contributions = []

    for w in words:
        score = get_word_score(
            w,
            global_lexicon,
            global_counts,
            user_lexicon,
            user_counts,
            global_mean,
            k
        )

        if score is not None:
            scores.append(score)
            contributions.append((w, score))

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