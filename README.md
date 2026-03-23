# MoodLens — ML-Powered Mental Health Journal & Analytics

A full-stack, machine learning-driven personal analytics platform. **MoodLens** goes beyond standard journaling by applying natural language processing (NLP), regression modeling, and Retrieval-Augmented Generation (RAG) to uncover hidden patterns in personal behavioral data and forecast future mood trajectories.

![MoodLens Dashboard](https://via.placeholder.com/800x400?text=MoodLens+Dashboard)

---

## 🚀 Technical Highlights for Recruiters

This project was built to demonstrate end-to-end ML engineering capabilities, specifically focusing on combining traditional statistical learning with modern LLM workflows in a production-ready web application.

- **Hybrid Bayesian NLP**: Designed a custom lexicon engine that learns user-specific word associations with mood, applying empirical Bayesian shrinkage to smooth predictions against a global population baseline.
- **Time Series Forecasting**: Engineered rolling, lagged, and cyclical temporal features (sin/cos encoding of 60-day cycles) to train a Multi-Output Ridge Regression model predicting mood 3, 7, and 14 days into the future.
- **RAG Architecture**: Integrated Hugging Face's `sentence-transformers` for dense vector embeddings of journal entries, performing cosine-similarity retrieval to ground an LLM (Mistral-7B via Inference API) in the user's historical context.
- **Production Infrastructure**: Built with Flask and SQLAlchemy, handling session caching, database abstraction (SQLite/PostgreSQL compatible via Supabase), and Dockerized for seamless Render deployment with pre-baked model weights to minimize cold-start latency.

---

## 🧠 Machine Learning Architecture & Mathematics

### 1. Hybrid Bayesian Lexicon (NLP)
Instead of relying purely on pre-trained sentiment models like VADER, the system builds its own word-to-mood association dictionary from journal history.

For each word $w$, a **centered mood score** is computed against the corpus mean $\bar{\mu}$:
$$ \text{score}_{\text{global}}(w) = \frac{1}{|D_w|} \sum_{d \in D_w} \text{mood}_d - \bar{\mu} $$

To personalize the model without overfitting small user datasets, a **count-based shrinkage weight** $\lambda_w$ (with smoothing constant $k = 10$) acts as a James-Stein estimator, blending the user's specific vocabulary with the global prior:
$$ \lambda_w = \frac{n_u(w)}{n_u(w) + k} $$
$$ \text{score}_{\text{hybrid}}(w) = \lambda_w \cdot \text{score}_{\text{user}}(w) + (1 - \lambda_w) \cdot \text{score}_{\text{global}}(w) $$

When a user writes a new journal entry, the text is lemmatized via NLTK, and the hybrid scores of the constituent words are averaged to predict the absolute mood score on a 1–10 scale.

### 2. Multi-Horizon Mood Forecasting (Regression)
A Ridge regression model predicts rolling average mood over the next $h \in \{3, 7, 14\}$ days.

**Feature Engineering:**
At each time step $t$, the system extracts:
- Lags: $m_{t-1}, m_{t-2}$
- Rolling Averages: $\frac{1}{w}\sum_{i=0}^{w-1} m_{t-i}$ for $w \in \{3, 7, 14\}$
- Cyclical Time Encodings: $\sin(\frac{2\pi \cdot t}{60})$ and $\cos(\frac{2\pi \cdot t}{60})$ to capture weekly/monthly periodicity.
- Text Signal: The output scalar of the NLP Lexicon model.

**Model:**
Ridge regression is trained jointly on multi-output targets $Y \in \mathbb{R}^{N \times 3}$:
$$ \hat{Y} = X\hat{B}, \quad \hat{B} = \arg\min_B \|Y - XB\|_F^2 + \alpha \|B\|_F^2 $$

### 3. Retrieval-Augmented Generation (RAG)
To provide an interactive "AI Therapist" experience, the system uses semantic search over the user's journal history.
- **Embedding Generation**: Uses `all-MiniLM-L6-v2` to embed journal text into 384-dimensional dense vectors upon submission.
- **Retrieval**: Computes dot-product (cosine similarity on normalized vectors) to find the top $K$ most contextually relevant past entries.
- **Generation**: Formats a zero-shot prompt injecting the retrieved context into Mistral-7B (via Hugging Face API) to generate grounded, personalized insights.

---

## 🛠️ Tech Stack & Implementation Details

| Component | Technologies Used |
|---|---|
| **Backend Framework** | Python 3.12, Flask, Gunicorn |
| **Database ORM** | SQLAlchemy (Compatible with SQLite, PostgreSQL/Supabase) |
| **Machine Learning** | scikit-learn (Ridge Regression), pandas, numpy, joblib |
| **NLP & LLMs** | `sentence-transformers`, `huggingface-hub`, NLTK, VADER |
| **Frontend UI** | HTML5, Vanilla CSS (Custom Card Design System), Jinja2, Chart.js |
| **Deployment** | Docker, Render |

### Codebase Organization

- `models/` - Core mathematical logic (Feature Builder, Lexicon engine, Forecasting mathematics)
- `services/` - Abstraction layer handling business logic, database transactions, and model inference (`rag_service.py`, `lexicon_service.py`, etc.)
- `training/` - Offline scripts for extracting data, training the Ridge models, generating the hybrid lexicon, and exporting `joblib` artifacts.
- `main.py` - Flask routing and API endpoint definitions.

---

## ⚙️ Running the Project Locally

### Prerequisites
- Python 3.12+

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd journal_mvp

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Create an environment file
touch .env
```

Add your API keys to the `.env` file (the app defaults to local SQLite if no DB URL is provided):
```env
# Optional: Set to your Supabase PostgreSQL connection string
# DATABASE_URL=postgresql://...

# Required for RAG Chat: Your Hugging Face Inference API token
HUGGINGFACE_API_KEY=hf_your_token_here

# Required for Flask Sessions
SECRET_KEY=your-random-secret-key-123
```

### Run
```bash
python main.py
```
Access the application at `http://127.0.0.1:5000`.

---

## 🚢 Production Deployment

The repository is configured for immediate deployment via Render. No secrets or model artifacts are tracked in version control (`.gitignore` is strictly enforced). 

To deploy using Docker on Render:
1. Connect the GitHub repository to Render as a **Docker** deployment.
2. The provided `Dockerfile` will automatically pull the `python:3.12-slim` image, install dependencies, and pre-bake the Hugging Face sentence-transformer model to prevent cold-start timeouts.
3. Add the `DATABASE_URL` (Supabase), `HUGGINGFACE_API_KEY`, and `SECRET_KEY` into Render's Environment Variables dashboard.
4. Deploy. The app runs via `gunicorn main:app --bind 0.0.0.0:5000`.
