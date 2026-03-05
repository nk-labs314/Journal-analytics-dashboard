# Mental Health Journal Tracker — MVP

A Flask-based personal mental health journaling app that tracks mood, sleep, activity, and social interactions. Three distinct ML/NLP systems analyse the data and forecast future mood states.

---

## What It Does

Users log daily journal entries alongside behavioural data. The system runs three analytical layers in parallel:

1. **Hybrid Bayesian Lexicon** — personalised word-mood association model with global prior smoothing
2. **Multi-Horizon Mood Forecasting** — Ridge regression over lag/rolling/cyclical features predicting mood at t+3, t+7, t+14
3. **Delta Prediction** — TF-IDF + Ridge and lexicon-based approaches for predicting mood *change* rather than absolute mood

---

## Project Structure

```
clean_journal_mvp/
├── main.py                        # Flask app and routes
├── config.py                      # Env-driven config
├── requirements.txt
│
├── models/
│   ├── feature_builder.py         # Shared feature engineering (lag, rolling, sin/cos)
│   ├── forecasting.py             # Multi-horizon training logic
│   └── lexicon_model.py           # Hybrid global+user lexicon engine
│
├── services/
│   ├── analytics_service.py       # Dashboard computation pipeline
│   ├── data_service.py            # DB read/write layer (SQLAlchemy)
│   ├── forecast_service.py        # Model loading and live inference
│   └── insight_service.py         # Sentiment, trend detection, behaviour alerts
│
├── training/
│   └── train_forecast.py          # Ridge multi-output training + artifact export
│
├── evaluation/
│   ├── evaluate_forecasting.py    # MAE vs rolling baseline per horizon
│   ├── evaluate_lexicon.py        # MAE, R², Pearson r for lexicon model
│   └── evaluate_delta.py          # TF-IDF vs lexicon delta prediction comparison
│
├── dataset_generation/
│   ├── synthetic_dataset.py       # AR(1) + seasonal + shock mood simulator
│   └── data_generator.py
│
├── artifacts/
│   └── ridge_multi_output.pkl     # Serialised trained model
│
├── data/
│   └── synthetic_dataset_v{1,2,3}.csv
│
├── utils/
│   └── experiment_logger.py       # CSV-based experiment tracker
│
├── templates/                     # Jinja2 HTML templates
└── static/js/charts.js            # Chart.js visualisations
```

---

## ML Systems

### 1. Hybrid Bayesian Lexicon (`models/lexicon_model.py`)

Builds a word-to-mood association dictionary from journal history, blending a population-level global lexicon with a per-user personal lexicon.

#### Global Lexicon

For each word $w$ observed across all users, compute a **centred mood score** using the corpus mean $\bar{\mu}$:

$$
\text{score}_{\text{global}}(w) = \mathbb{E}[\text{mood} \mid w] - \bar{\mu} = \frac{1}{|D_w|} \sum_{d \in D_w} \text{mood}_d - \bar{\mu}
$$

where $D_w$ is the set of journal entries containing $w$. Only words with $|D_w| \geq 5$ are retained (frequency threshold).

#### User Lexicon

The same construction is applied per-user, centred on the **user's own mean** $\bar{\mu}_u$:

$$
\text{score}_{\text{user}}(w) = \frac{1}{|D_{w,u}|} \sum_{d \in D_{w,u}} \text{mood}_d - \bar{\mu}_u
$$

Words require $|D_{w,u}| \geq 3$ to enter the user lexicon.

#### Hybrid Blending (Empirical Bayes Prior)

The global lexicon acts as a prior. The user lexicon is blended in via a **count-based shrinkage weight** $\lambda_w$, with smoothing constant $k = 10$:

$$
\lambda_w = \frac{n_u(w)}{n_u(w) + k}
$$

$$
\text{score}_{\text{hybrid}}(w) = \lambda_w \cdot \text{score}_{\text{user}}(w) + (1 - \lambda_w) \cdot \text{score}_{\text{global}}(w)
$$

When $n_u(w) \to 0$, the score collapses to the global prior. When $n_u(w) \gg k$, it trusts the user's personal signal. This is analogous to a James-Stein estimator pulling low-evidence estimates toward a shared mean.

#### Mood Prediction from Text

For a given journal entry tokenised to words $\{w_1, \ldots, w_T\}$:

$$
\hat{\text{mood}} = \bar{\mu} + \frac{1}{|S|} \sum_{w \in S} \text{score}_{\text{hybrid}}(w)
$$

where $S \subseteq \{w_1, \ldots, w_T\}$ is the subset of words present in the global lexicon. The prediction is an absolute mood score on the 1–10 scale.

---

### 2. Multi-Horizon Mood Forecasting (`models/forecasting.py`, `training/train_forecast.py`)

Predicts rolling average mood over the next $h \in \{3, 7, 14\}$ days using a Ridge multi-output regression.

#### Feature Engineering

For each user's chronological mood sequence, the following features are constructed at each time step $t$:

| Feature | Formula |
|---|---|
| `lag_1` | $m_{t-1}$ |
| `lag_2` | $m_{t-2}$ |
| `rolling_3` | $\frac{1}{3}\sum_{i=0}^{2} m_{t-i}$ |
| `rolling_7` | $\frac{1}{7}\sum_{i=0}^{6} m_{t-i}$ |
| `rolling_14` | $\frac{1}{14}\sum_{i=0}^{13} m_{t-i}$ |
| `sin_time` | $\sin\!\left(\frac{2\pi \cdot t}{60}\right)$ |
| `cos_time` | $\cos\!\left(\frac{2\pi \cdot t}{60}\right)$ |

The sin/cos pair encodes a 60-day cycle to capture weekly/monthly mood periodicity without requiring an explicit date feature.

#### Target Construction

For each horizon $h$, the target is the **forward rolling average** — the mean mood over the next $h$ days:

$$
y_{t,h} = \frac{1}{h} \sum_{i=1}^{h} m_{t+i}
$$

Implemented as a shifted rolling mean: `mood_score.shift(-1).rolling(window=h).mean()`.

#### Model

Ridge regression is trained with multi-output targets $Y \in \mathbb{R}^{N \times 3}$:

$$
\hat{Y} = X\hat{B}, \quad \hat{B} = \arg\min_B \|Y - XB\|_F^2 + \alpha \|B\|_F^2
$$

with $\alpha = 1.0$. A single model jointly predicts all three horizons. Users are processed independently to construct features, then pooled for training.

#### Baseline Comparison

At evaluation time, model MAE is compared to a naive rolling-7 baseline (no model, just use the current 7-day average as the forecast). Both are reported in `evaluate_forecasting.py`.

---

### 3. Delta Prediction (`evaluation/evaluate_delta.py`)

Rather than predicting absolute mood, this predicts **mood change** — the signed difference between current and future mood:

$$
\delta_{t,h} = m_{t+h} - m_t
$$

Two approaches are compared:

**TF-IDF + Ridge**: Journal text is vectorised (top 5000 features, $\ell_2$-normalised TF-IDF), then Ridge regression ($\alpha = 1.0$) is trained to predict $\delta$.

**Lexicon Delta**: The hybrid lexicon predicts a raw mood score from text. A linear regression is then fit on training data to map the lexicon's centred output to $\delta$:

$$
\hat{\delta} = \beta_0 + \beta_1 \cdot (\hat{\text{mood}}_{\text{lexicon}} - \bar{\mu})
$$

Both are evaluated against a zero-delta baseline (predict no change), which represents the hardest-to-beat naive baseline for delta tasks.

---

### 4. Synthetic Data Generation (`dataset_generation/synthetic_dataset.py`)

Training data is generated using a **mean-reverting AR(1) process with seasonal component and random shocks** — an Ornstein-Uhlenbeck-like discrete model:

$$
m_t = \mu + \phi(m_{t-1} - \mu) + A \sin\!\left(\frac{2\pi t}{T}\right) + \epsilon_t + s_t
$$

where:
- $\mu$ — user-specific baseline mood
- $\phi \in (0, 1)$ — mean-reversion strength (AR coefficient)
- $A$ — seasonal amplitude
- $T$ — cycle period (days)
- $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ — Gaussian noise
- $s_t \sim \text{Uniform}(-3, 3)$ with probability $p_{\text{shock}}$, else $0$

Four user regime types are sampled: `stable_high`, `volatile`, `low_mean`, `cyclical` — each with distinct $(\mu, \phi, \sigma, A, T, p_{\text{shock}})$ parameters.

Journal text is generated conditional on current mood, momentum, and streak (consecutive rising/falling days), injecting trend-aware vocabulary to create predictive signal in the text.

---

## Setup

### Prerequisites

- Python 3.12+

### Installation

```bash
git clone <your-repo-url>
cd clean_journal_mvp

pip install -r requirements.txt

# NLTK resources required by the lexicon model
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
"
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `dev_secret_key` | Flask session secret — **change in production** |
| `DATABASE_URL` | `sqlite:///mental_health.db` | SQLAlchemy DB connection string |
| `DEBUG` | `True` | Enables debug mode and DB auto-init |

### Run

```bash
python main.py
# → http://localhost:5000
```

---

## Experimental Results

All runs logged automatically to `experiments_log.csv`. Dataset: 10 users × 500 entries, 80/20 chronological split.

### Lexicon Model — `synthetic_dataset_v1`

| Metric | Hybrid Lexicon | Baseline (global mean) |
|---|---|---|
| MAE | **2.229** | 2.935 |
| R² | **0.371** | — |
| Pearson r | **0.949** | — |

The 0.949 correlation is high and expected given the synthetic data has a direct word→mood signal baked in. R² of 0.37 is more honest — it captures about a third of mood variance from text alone. The 24% MAE reduction over the global mean baseline confirms the personalisation is working.

### Multi-Horizon Forecasting — `synthetic_dataset_v1`

| Horizon | Model MAE | Baseline MAE (rolling-7) | Improvement |
|---|---|---|---|
| 3 days | **0.271** | 0.906 | 70% |
| 7 days | **0.147** | 0.282 | 48% |
| 14 days | **0.091** | 0.651 | 86% |

Longer horizons yield lower absolute MAE because the rolling-average target smooths out variance — harder numbers to interpret in isolation, but the model consistently outperforms the naive rolling baseline across all three.

### Delta Prediction (7-day) — Model Comparison Across Dataset Versions

Predicting signed mood change $\delta = m_{t+7} - m_t$ rather than absolute mood. Baseline: predict zero change.

| Dataset | Model | MAE | R² | Baseline MAE |
|---|---|---|---|---|
| v1 | TF-IDF + Ridge | 1.997 | 0.172 | 1.964 |
| v2 | TF-IDF + Ridge | 1.833 | 0.147 | 1.851 |
| v2 | Lexicon Delta | **1.814** | 0.143 | 1.851 |
| v3 | TF-IDF + Ridge | 1.853 | 0.157 | 1.938 |
| v3 | Lexicon Delta | 1.864 | 0.151 | 1.938 |

Delta prediction is the hardest task here — R² hovers around 0.14–0.17, and v1 TF-IDF barely beats the zero-change baseline. This is expected: short-term mood change is largely driven by external events not captured in a single journal entry. The marginal lexicon advantage on v2 disappears on v3, suggesting it's not robust across data regimes. Worth noting — not hiding.

---

## Training & Evaluation

```bash
# Retrain the Ridge forecast model
python training/train_forecast.py

# Evaluate forecasting MAE vs rolling baseline
python evaluation/evaluate_forecasting.py

# Evaluate lexicon model (MAE, R², Pearson r)
python evaluation/evaluate_lexicon.py

# Compare TF-IDF vs lexicon on delta prediction
python evaluation/evaluate_delta.py
```

All runs are logged automatically to `experiments_log.csv` via `utils/experiment_logger.py`.

---

## Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Home / journal entry form |
| `/log` | POST | Submit mood + behaviour log |
| `/dashboard` | GET | Analytics dashboard |
| `/journals` | GET | Past journal entries |
| `/forecast` | GET | 3/7/14-day mood forecast |
| `/health` | GET | Health check (DB + model status) |

---

## Known Limitations / Planned Work

- [ ] Authentication — currently single hardcoded `user_id=1`
- [ ] Chatbot module — scaffolded in `main.py`, not yet integrated
- [ ] SQLite not suitable for multi-user deployment — use `DATABASE_URL` to swap to PostgreSQL
- [ ] Lexicon model rebuilds from DB on every request — should be cached
- [ ] Add `python-dotenv` for `.env` file support
- [ ] Lexicon cold-start: new users with no history fall back entirely to global prior

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Flask |
| Database | SQLite via SQLAlchemy |
| ML | scikit-learn (Ridge, LinearRegression, TF-IDF) |
| NLP | NLTK (tokenisation, lemmatisation), VADER Sentiment |
| Visualisation | Chart.js |
| Model serialisation | joblib |
