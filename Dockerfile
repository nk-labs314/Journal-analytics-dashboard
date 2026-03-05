FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformer model during build
# so cold starts are fast (no download delay on startup)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Pre-download NLTK data to prevent runtime download corruption (BadZipFile)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
COPY . .

EXPOSE 5000
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:5000", "--workers", "2"]
