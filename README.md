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

An AI-powered journaling system that analyzes behavioral patterns, predicts future mood trends, and generates personalized insights using a Retrieval-Augmented Generation (RAG) pipeline.

This project combines traditional machine learning (regression, feature engineering) with modern LLM-based reasoning to build a complete end-to-end analytics system.

---

## Features

* User Authentication
  Secure session-based login system built with Flask

* Journal Logging
  Tracks mood, behavioral signals (sleep, activity, social interaction), and text entries

* Analytics Dashboard
  Computes trends, correlations, and behavioral insights

* Forecasting Engine
  Predicts future mood using time-series features and regression

* Insight Engine
  Custom lexicon-based NLP system with user-specific adaptation

* RAG-based Chat Assistant
  Retrieves relevant past entries and generates grounded responses using an LLM

---

## System Architecture

```
Frontend (HTML Templates)
        ↓
Flask Backend (Routes + Services)
        ↓
-----------------------------------
| Supabase (PostgreSQL Database) |
-----------------------------------
        ↓
-----------------------------------
| Embeddings: SentenceTransformers |
| LLM: OpenRouter (GPT-3.5 Turbo)  |
-----------------------------------
```

---

## Architecture Diagram

```
                ┌────────────────────┐
                │     Frontend       │
                │ (HTML Templates)   │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │    Flask Backend   │
                │  (Routes + Logic)  │
                └─────────┬──────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌──────────────┐  ┌────────────────┐  ┌────────────────────┐
│  Supabase DB │  │ Embedding Model│  │   OpenRouter LLM   │
│ (PostgreSQL) │  │ MiniLM-L6-v2   │  │   GPT-3.5 Turbo    │
└──────────────┘  └────────────────┘  └────────────────────┘
        │
        ▼
┌────────────────────┐
│ Stored Embeddings  │
│ + Journal Entries  │
└────────────────────┘
```

---

## RAG Pipeline (How it Works)

1. User submits a query
2. Query is converted into an embedding vector
3. System retrieves top-k similar journal entries
4. Retrieved entries + analytics + forecast are combined into context
5. LLM generates a grounded, personalized response

This ensures:

* responses are based on user history
* reduced hallucination
* context-aware insights

---

## Machine Learning Design

### 1. Hybrid Lexicon-Based NLP

The system builds a custom word-to-mood mapping.

For each word w:

score_global(w) = average mood of entries containing w − global mean

To avoid overfitting, a shrinkage factor is applied:

λ_w = n_u(w) / (n_u(w) + k)

Final score:

score_hybrid(w) = λ_w · score_user(w) + (1 − λ_w) · score_global(w)

This allows:
- personalization of vocabulary  
- stability for low-frequency words  

### 2. Mood Forecasting (Ridge Regression)

Future mood is predicted using a multi-output regression model:

Ŷ = X·B

The model is trained using ridge regularization:

min ||Y - X·B||² + α||B||²

Features include:
- lag values (m_t-1, m_t-2)
- rolling averages (3, 7, 14 days)
- cyclical encoding (sin, cos)
- NLP-derived sentiment signals

This design:
- handles noisy behavioral data  
- avoids overfitting  
- enables multi-horizon forecasting  

### 3. Retrieval-Augmented Generation (RAG)

* Embeddings generated using MiniLM (384-d vectors)
* Cosine similarity used for retrieval
* Context injected into LLM prompt

This ensures:

* personalized responses
* grounding in historical data
* reduced hallucination

---

## Tech Stack

* Backend: Flask, SQLAlchemy
* Database: Supabase (PostgreSQL)
* Machine Learning:

  * scikit-learn (Ridge Regression)
  * SentenceTransformers (embeddings)
* LLM: OpenRouter (GPT-3.5 Turbo)
* Deployment: Hugging Face Spaces (Docker)

---

## Code Structure

```
main.py        → Flask routes and entry point  
services/      → business logic (RAG, analytics, auth, embeddings)  
models/        → ML logic (forecasting, lexicon scoring)  
training/      → offline model training scripts  
```

---

## Key Design Decisions

### Custom Backend over BaaS

A Flask backend was implemented to:

* maintain full control over logic
* demonstrate backend engineering capability

---

### Hybrid AI Architecture

* Embeddings computed locally
* LLM handled via external API

This balances:

* cost
* performance
* reliability

---

### RAG for Personalization

Combines:

* retrieved journal entries
* analytics signals
* forecast outputs

into a unified prompt for the LLM

---

## Key Engineering Challenges

### Session Persistence in Deployment

Resolved cookie/session issues in a proxied environment (Hugging Face Spaces).

---

### LLM Provider Limitations

Switched from Hugging Face inference APIs to OpenRouter due to model/provider constraints.

---

### Context Construction for RAG

Designed a structured pipeline combining multiple data sources into a coherent prompt.

---

## Results

* Retrieves relevant historical entries using embeddings
* Generates context-aware, grounded responses
* Maintains user-level data isolation via backend logic
* End-to-end system functioning across multiple components

---

## Setup

```bash
git clone https://github.com/nk-labs314/Journal-analytics-dashboard.git
cd Journal-analytics-dashboard
pip install -r requirements.txt
```

---

## Environment Variables

```
SECRET_KEY=your_secret_key
OPENROUTER_API_KEY=your_api_key
DATABASE_URL=your_supabase_database_url
```

---

## Run Locally

```bash
python main.py
```

---

## Future Improvements

* Add RLS + Supabase Auth integration
* Improve retrieval ranking in RAG pipeline
* Add caching for LLM responses
* Scale backend architecture

---

## Author

Nandan Kailasanath
