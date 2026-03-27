---
title: Journal Analytics Dashboard
emoji: 📔
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
---
# Journal-Analytics-Dashboard

A full-stack, machine learning-driven personal analytics platform. Journal Analytics goes beyond standard journaling by applying natural language processing, regression modeling, and Retrieval-Augmented Generation to uncover patterns in personal behavioral data and forecast future mood trajectories.

Deployed on Hugging Face Spaces: https://huggingface.co/spaces/MetHJ/journal-analytics-dashboard

---

## Technical Highlights

This project demonstrates end-to-end ML engineering — combining traditional statistical learning with modern LLM workflows in a production web application.

- **Hybrid Bayesian NLP**: A custom lexicon engine that learns user-specific word-to-mood associations. It uses empirical Bayesian shrinkage to blend personal vocabulary against a global population baseline.
- **Time Series Forecasting**: Rolling, lagged, and cyclical temporal features (sin/cos encoding of 60-day cycles) feed a Multi-Output Ridge Regression model predicting mood at 3, 7, and 14-day horizons.
- **RAG Architecture**: Hugging Face `sentence-transformers` generate dense vector embeddings of journal entries. Cosine similarity retrieval grounds an LLM (GPT-3.5 Turbo via OpenRouter) in the user's historical context.
- **Production Infrastructure**: Flask and SQLAlchemy handle session management and database abstraction (SQLite locally, PostgreSQL/Supabase in production). The app is Dockerized for Hugging Face Spaces deployment.
- **Authentication & Security**: We fixed session handling to use sliding expiration windows, added a strict CORS policy, and built a dedicated `/settings` route for password updates and destructive account deletions. We also standardized the auth workflow with strict tuple error-handling.

---

## System Architecture

```text
                +----------------------+
                |      Frontend        |
                |  (HTML Templates)    |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |    Flask Backend     |
                |  (Routes + Logic)    |
                +----------+-----------+
                           |
        +------------------+-------------------+
        v                  v                   v
+--------------+  +----------------+  +--------------------+
|  Supabase DB |  | Embedding Model|  |   OpenRouter LLM   |
| (PostgreSQL) |  | MiniLM-L6-v2   |  |   GPT-3.5 Turbo    |
+--------------+  +----------------+  +--------------------+
        |
        v
+----------------------+
| Stored Embeddings    |
| + Journal Entries    |
+----------------------+
```

---

## Machine Learning Architecture

### 1. Hybrid Bayesian Lexicon (NLP)

Instead of relying purely on pre-trained sentiment models like VADER, the system builds its own word-to-mood association dictionary from journal history.

For each word $w$, a centered mood score is computed against the corpus mean $\bar{\mu}$:

$$\text{score}_{\text{global}}(w) = \frac{1}{|D_w|} \sum_{d \in D_w} \text{mood}_d - \bar{\mu}$$

To personalize without overfitting small user datasets, a count-based shrinkage weight $\lambda_w$ (smoothing constant $k = 10$) blends the user's vocabulary against the global prior:

$$\lambda_w = \frac{n_u(w)}{n_u(w) + k}$$

$$\text{score}_{\text{hybrid}}(w) = \lambda_w \cdot \text{score}_{\text{user}}(w) + (1 - \lambda_w) \cdot \text{score}_{\text{global}}(w)$$

When a user writes a new journal entry, the text is lemmatized via NLTK, and the hybrid scores of the constituent words are averaged to predict mood on a 1-10 scale.

### 2. Multi-Horizon Mood Forecasting (Regression)

A Ridge regression model predicts rolling average mood over the next $h \in \{3, 7, 14\}$ days.

Feature engineering at each time step $t$:

- Lags: $m_{t-1},\ m_{t-2}$
- Rolling averages: $\frac{1}{w}\sum_{i=0}^{w-1} m_{t-i}$ for $w \in \{3, 7, 14\}$
- Cyclical time encodings: $\sin\!\left(\frac{2\pi \cdot t}{60}\right)$ and $\cos\!\left(\frac{2\pi \cdot t}{60}\right)$
- Text signal: scalar output from the NLP lexicon model

Ridge regression is trained jointly on multi-output targets $Y \in \mathbb{R}^{N \times 3}$:

$$\hat{Y} = X\hat{B}, \quad \hat{B} = \arg\min_B \|Y - XB\|_F^2 + \alpha\|B\|_F^2$$

### 3. RAG Pipeline

To provide an AI assistant grounded in the user's history, the system uses semantic search over journal entries.

1. The user submits a query.
2. The query is converted into a 384-dimensional embedding vector via `all-MiniLM-L6-v2`.
3. Dot-product similarity (cosine on normalized vectors) retrieves the top-K most relevant past entries.
4. Retrieved entries, current analytics, and mood forecast form a structured context block.
5. GPT-3.5 Turbo (via OpenRouter) generates a grounded response.

This ensures responses are anchored to user history, reducing hallucination and keeping insights specific to the individual.

---

## Tech Stack

| Component | Technologies |
|---|---|
| Backend Framework | Python 3.12, Flask, Gunicorn |
| Database ORM | SQLAlchemy (SQLite locally, PostgreSQL/Supabase in production) |
| Machine Learning | scikit-learn (Ridge Regression), pandas, numpy, joblib |
| NLP and Embeddings | sentence-transformers, NLTK, VADER |
| LLM | GPT-3.5 Turbo via OpenRouter |
| Frontend | HTML5, Vanilla CSS, Jinja2, Chart.js |
| Deployment | Docker, Hugging Face Spaces |

---

## Codebase Structure

```text
main.py                  Flask routes and entry point
config.py                Environment and app configuration
services/
    rag_service.py       Retrieval and LLM generation
    embedding_service.py Sentence-transformer wrapper
    lexicon_service.py   Lexicon loading and text analysis
    analytics_service.py Dashboard computation
    forecast_service.py  Model inference wrapper
    data_service.py      Database queries
    auth_service.py      User creation and verification
models/
    lexicon_model.py     Bayesian lexicon math
    forecasting.py       Feature extraction logic
    feature_builder.py   Time-series feature engineering
training/
    train_forecast.py    Offline Ridge model training
    train_lexicon.py     Offline global lexicon training
artifacts/
    ridge_multi_output.pkl  Trained Ridge model + lexicon bundle
```

---

## Key Design Decisions

**Custom backend over BaaS**: A Flask backend gives us full control over inference logic, session handling, and the RAG pipeline.

**Hybrid AI architecture**: Embeddings are computed locally using sentence-transformers. LLM inference goes through the OpenRouter API. This splits the load — compute-heavy embedding stays in-process, while LLM calls scale externally.

**RAG for personalization**: Combining retrieved journal entries, live analytics signals, and forecast outputs into a single prompt forces the LLM to anchor its answers in the user's actual data.

---

## Engineering Fixes in This Build

**Session persistence**: Running behind a reverse proxy on Hugging Face Spaces broke cookies. We fixed this with explicit `SameSite=None; Secure=True` settings and a rolling window for active sessions.

**Auth Hardening**: Route error unwrapping caused critical failures during login. We standardized all auth returns to a clean 3-tuple `(user_id, auth_hash, error)` and added CORS enforcement everywhere. Built out a robust E2E test suite.

**LLM optimization**: Switched from Hugging Face Inference API to OpenRouter to bypass severe rate limit constraints on free serverless endpoints.

---

## Running Locally

### Prerequisites

Python 3.12+

### Setup

```bash
git clone https://github.com/nk-labs314/Journal-analytics-dashboard.git
cd Journal-analytics-dashboard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
touch .env
```

Add to `.env`:

```text
SECRET_KEY=your-random-secret-key
OPENROUTER_API_KEY=your-openrouter-api-key
# Optional: Supabase PostgreSQL connection string
# DATABASE_URL=postgresql://...
```

### Run

```bash
python main.py
```

Access at `http://127.0.0.1:5000`.

---

## Production Deployment

The app is deployed on Hugging Face Spaces as a containerized Flask service.

The `Dockerfile` pulls `python:3.12-slim`, installs dependencies, and pre-bakes the `all-MiniLM-L6-v2` sentence-transformer model to avoid cold starts.

Required environment variables in the Spaces settings: `SECRET_KEY`, `OPENROUTER_API_KEY`, and optionally `DATABASE_URL` for Supabase PostgreSQL.

The app runs via: `gunicorn main:app --bind 0.0.0.0:7860`

---

## Planned Improvements

- CSRF protection on all form endpoints
- Supabase Auth integration with Row Level Security
- Retraining pipeline on real user data (currently trains on synthetic data)
- Persistent multi-turn chat history
- LLM response caching for repeated queries
- Improved retrieval ranking in the RAG pipeline
