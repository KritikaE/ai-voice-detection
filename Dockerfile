# ---- Base image ----
FROM python:3.10-slim

# ---- Environment ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- System deps for librosa ----
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory ----
WORKDIR /app

# ---- Install Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy source ----
COPY app/ app/
COPY model/ model/

# ---- Expose port ----
EXPOSE 8080

# ---- Start server ----
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
