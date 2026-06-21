FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY gtzan_selected_features.csv .

RUN python3 app/train_and_save.py

EXPOSE 7860

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
