---
title: Journal Analytics Dashboard
emoji: 📔
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Journal Analytics Dashboard

A personal analytics platform that applies NLP, time-series regression, and retrieval-augmented generation to journal entries — turning freeform text into mood forecasts and grounded AI insights.

*Important:** Open in a normal browser tab.  
The embedded preview (especially in incognito/private mode) may not work due to browser cookie restrictions.

**Live demo:** https://huggingface.co/spaces/MetHJ/journal-analytics-dashboard  
**Source:** https://github.com/nk-labs314/Journal-analytics-dashboard

**!!Note on HuggingFace Spaces Preview**

The app may not function correctly in the embedded preview (especially in incognito/private mode) due to browser restrictions on third-party cookies.
This project uses session-based authentication with CSRF protection, which relies on cookies. Some browsers block these cookies in cross-site iframe contexts.

To use the app reliably, open the Space in a new tab.
---

## What it does

You write a journal entry. The system scores it with a personalized Bayesian lexicon, stores a dense vector embedding of it, updates your mood time series, and runs a multi-horizon Ridge regression to forecast your mood over the next 3, 7, and 14 days. When you open the chat, a RAG pipeline retrieves your most relevant past entries using cosine similarity and passes them — alongside your live analytics and forecast — to an LLM, so the responses are grounded in your actual history rather than generic advice.

---

## System Architecture

```text

                +----------------------+
                |      Frontend        |
                | (Jinja + Chart.js)   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |    Flask Backend     |
                | (Routes + Validation)|
                +----------+-----------+
                           |
        +------------------+-------------------+
        |                  |                   |
        v                  v                   v

+----------------+  +----------------+  +----------------------+
|  Data Service  |  | Analytics Svc  |  |   Forecast Service   |
| (SQLAlchemy)   |  | (Trends, Corr) |  |   (Ridge Model)      |
+--------+-------+  +--------+-------+  +----------+-----------+
         |                   |                      |
         v                   v                      v
+-----------------------------------------------------------+
|                       Database                            |
| MoodLogs | BehaviorData | EntryEmbeddings | AuthUsers     |
+-----------------------------------------------------------+

                           |
                           v
                +----------------------+
                |  Embedding Service   |
                | (MiniLM-L6-v2)       |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |     RAG Service      |
                | Retrieval + Context  |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |   OpenRouter LLM     |
                |   (GPT-3.5-turbo)    |
                +----------------------+
```
```
RAG Flow:

Query
 → Embedding
 → Similarity Search (EntryEmbeddings)
 → Top-K Retrieval
 → + Analytics + Forecast
 → LLM Response
 ```
---

## ML Components

### 1. Hybrid Bayesian Lexicon

Rather than using a pre-trained sentiment model, the system builds a word-to-mood dictionary from journal history and personalizes it per user via empirical Bayesian shrinkage.

For each word $w$, a centered score is computed relative to the corpus mean $\bar{\mu}$:

$$\text{score}_{\text{global}}(w) = \frac{1}{|D_w|} \sum_{d \in D_w} \text{mood}_d - \bar{\mu}$$

A count-based shrinkage weight $\lambda_w$ (smoothing constant $k = 10$) blends the user's vocabulary against the global prior, avoiding overfitting on small per-user datasets:

$$\lambda_w = \frac{n_u(w)}{n_u(w) + k}$$

$$\text{score}_{\text{hybrid}}(w) = \lambda_w \cdot \text{score}_{\text{user}}(w) + (1 - \lambda_w) \cdot \text{score}_{\text{global}}(w)$$

The tokenizer lemmatizes via NLTK, removes stopwords while preserving negation words (`not`, `never`, `can't`, etc.), and applies a 2-token look-back window to flip scores on negated words (`"not happy"` → negated). High-valence terms (`suicidal`, `hopeless`, `productive`) have hard overrides to avoid being diluted by context.

### 2. Multi-Horizon Mood Forecasting (Ridge Regression)

A Ridge regression model predicts rolling average mood over the next $h \in \{3, 7, 14\}$ days.

Feature set at each time step $t$:

- Lags: $m_{t-1},\ m_{t-2}$
- Rolling averages: $\frac{1}{w}\sum_{i=0}^{w-1} m_{t-i}$ for $w \in \{3, 7, 14\}$
- Cyclical time encodings: $\sin\!\left(\frac{2\pi \cdot t}{60}\right)$ and $\cos\!\left(\frac{2\pi \cdot t}{60}\right)$
- Lexicon signal: centered score from the per-entry NLP model (bridges text signal into the time-series model)

The model is trained jointly on all three horizons as multi-output targets $Y \in \mathbb{R}^{N \times 3}$:

$$\hat{Y} = X\hat{B}, \quad \hat{B} = \arg\min_B \|Y - XB\|_F^2 + \alpha\|B\|_F^2$$

with $\alpha = 1.0$. Users are processed independently (no cross-user leakage), and targets are built from rolling averages of forward-shifted mood scores.

### 3. Delta Forecasting (Mood Change Prediction)

A separate evaluation track benchmarks predicting the mood *change* $\delta = m_{t+h} - m_t$ rather than the absolute value. Two approaches are compared:

- **TF-IDF + Ridge:** 5,000-feature TF-IDF representation of journal text → Ridge regression on delta targets
- **Lexicon Delta:** centered lexicon score → linear calibration layer trained to map lexicon signal to delta

Both are evaluated against a zero-delta baseline (predicting no change) and logged via the experiment tracker.

### 4. RAG Pipeline

1. User submits a query.
2. The query is embedded into 384 dimensions via `all-MiniLM-L6-v2`.
3. Stored entry embeddings (serialized as raw float32 bytes in PostgreSQL) are deserialized and scored via dot-product similarity on normalized vectors.
4. Top-K retrieved entries, current analytics summary, and mood forecast form a structured context block.
5. The LLM generates a response anchored to that context, with a strict system prompt that refuses off-topic queries.

Embeddings are preloaded at startup to avoid cold-start latency on Hugging Face Spaces.

---

## Evaluation Infrastructure

Three independent evaluation scripts with chronological train/test splits (no future leakage) and a persistent CSV experiment logger:

| Script | Model | Metrics logged |
|---|---|---|
| `evaluate_lexicon.py` | Hybrid Bayesian lexicon | MAE, R², Pearson r vs mean baseline |
| `evaluate_forecasting.py` | Multi-output Ridge (3/7/14-day) | MAE per horizon vs lag-1 baseline |
| `evaluate_delta.py` | TF-IDF Ridge + Lexicon Delta | MAE, R² vs zero-delta baseline |

The experiment logger (`utils/experiment_logger.py`) records dataset version, model name, hyperparameters, metrics, baseline metrics, and a timestamp to `experiments_log.csv` — so benchmark numbers are reproducible and version-tracked.

---

## Tech Stack

| Component | Technologies |
|---|---|
| Backend | Python 3.12, Flask (App Factory), Gunicorn |
| Database ORM | SQLAlchemy (SQLite locally, PostgreSQL/Supabase in production) |
| Machine Learning | scikit-learn (Ridge, LinearRegression, TF-IDF), pandas, numpy, joblib |
| NLP and Embeddings | sentence-transformers (MiniLM-L6-v2), NLTK (lemmatizer, negation-aware tokenizer), VADER |
| LLM | GPT-3.5-turbo via OpenRouter |
| Frontend | HTML5, Vanilla CSS, Jinja2, Chart.js, Lucide SVG icons |
| Deployment | Docker, Hugging Face Spaces |

---

## Codebase Structure

```
main.py                      Flask app factory and route registration
config.py                    Environment config and app settings
models/
    lexicon_model.py         Bayesian shrinkage lexicon — training, scoring, negation handling
    forecasting.py           Time-series feature construction and per-horizon train/eval
    feature_builder.py       Feature engineering — lags, rolling windows, cyclical encoding, lexicon score
training/
    train_forecast.py        Offline Ridge training — serializes model + lexicon into a single artifact
    train_lexicon.py         Offline global lexicon training
evaluation/
    evaluate_lexicon.py      Lexicon benchmark with experiment logging
    evaluate_forecasting.py  Forecasting benchmark with experiment logging
    evaluate_delta.py        Delta prediction — TF-IDF Ridge vs lexicon comparison
utils/
    experiment_logger.py     Append-only CSV experiment tracker with UUID-stamped runs
services/
    auth_service.py          Session management, sliding expiry, password and account handling
    rag_service.py           Vector retrieval + LLM generation with conversation memory
    embedding_service.py     Sentence-transformer wrapper with similarity scoring
    lexicon_service.py       Lexicon loading and inference at request time
    analytics_service.py     Dashboard aggregates and mood trend computation
    forecast_service.py      Inference wrapper for the trained Ridge artifact
    data_service.py          SQLAlchemy query layer
    demo_service.py          Demo account seeding — backdated entries with real MiniLM embeddings
artifacts/
    ridge_multi_output.pkl   Trained Ridge model + bundled global lexicon
    global_lexicon.pkl       Standalone lexicon artifact
```

---

## Key Design Decisions

**Why a custom lexicon over VADER?** VADER uses a static, domain-general word list. Personal journals have idiosyncratic vocabulary — the word "run" means something different to a runner than to someone describing emotional avoidance. The Bayesian shrinkage approach learns user-specific associations while falling back to the population signal for rare words.

**Why Ridge over a neural sequence model?** With a realistic 30–100 entries per user, a transformer would overfit immediately. Ridge with $\ell_2$ regularization and explicit temporal features is more interpretable and actually fits the data size.

**Why split embeddings from LLM inference?** Sentence-transformer inference runs in-process (no API latency, no cost per call). LLM inference goes through OpenRouter. Splitting these lets the retrieval step stay fast while the generation step stays scalable.

**Why a delta evaluation track?** Absolute mood prediction conflates regression-to-the-mean with genuine predictive signal. Predicting the change from current mood to future mood is a harder and more honest benchmark.

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
```

NLTK data (first run only):
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

Create `.env`:
```
SECRET_KEY=your-random-secret-key
OPENROUTER_API_KEY=your-openrouter-api-key
# Optional — Supabase PostgreSQL. Omit for local SQLite.
# DATABASE_URL=postgresql://...
```

### Run

```bash
python main.py
```

Available at `http://127.0.0.1:5000`.

### Run evaluations

```bash
cd evaluation
python evaluate_lexicon.py      # Bayesian lexicon benchmark
python evaluate_forecasting.py  # Ridge forecasting benchmark
python evaluate_delta.py        # Delta prediction comparison
```

Results are appended to `experiments_log.csv`.

---

## Production Deployment

Deployed on Hugging Face Spaces as a containerized Flask service.

The `Dockerfile` pulls `python:3.12-slim`, installs dependencies, and pre-bakes `all-MiniLM-L6-v2` to avoid cold starts. The app runs via Gunicorn: `gunicorn main:app --bind 0.0.0.0:7860`.

CSRF is handled via a custom `before_request` hook — `secrets.token_hex(32)` stored in session, validated on all mutating requests against both the form field and `X-CSRFToken` header (for AJAX calls).

Required Spaces secrets: `SECRET_KEY`, `OPENROUTER_API_KEY`. For Supabase: `DATABASE_URL` (note: no trailing whitespace — the PostgreSQL driver will reject the connection silently).

---

## Known Limitations

- Training data is synthetic. The ML pipeline works end-to-end, but benchmark numbers reflect synthetic distributions rather than real user behavior.
- Chat history is not persisted across sessions.
- Retrieval ranking uses dot-product similarity — no reranking step.

---

## Planned

- Retrain on real user data once sufficient entries are collected
- Supabase Auth with Row Level Security
- Persistent multi-turn chat history
- LLM response caching for repeated queries
- Reranker on top of the retrieval step
