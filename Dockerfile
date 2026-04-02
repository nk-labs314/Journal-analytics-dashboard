FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN python - <<EOF
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
EOF

COPY main.py config.py ./
COPY services/ ./services/
COPY templates/ ./templates/
COPY static/ ./static/
COPY artifacts/ ./artifacts/

EXPOSE 7860

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "main:app"]