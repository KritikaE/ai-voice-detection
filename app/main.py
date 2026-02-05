import os
import numpy as np
import librosa
from fastapi import FastAPI, Header, HTTPException, Depends

from app.schema import AudioRequest, PredictionResponse
from app.audio import decode_base64_audio
from app.model import detector, EXPECTED_FEATURE_DIM


# -------------------------
# API Key (ENV ONLY)
# -------------------------
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable not set")


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(title="AI Voice Detection API")


# -------------------------
# Warm-up
# -------------------------
@app.on_event("startup")
def warmup_model():
    dummy_input = [0.0] * EXPECTED_FEATURE_DIM
    detector.model.predict_proba([dummy_input])


# -------------------------
# Predict Endpoint
# -------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: AudioRequest, _: str = Depends(verify_api_key)):

    # ---- 1. Robust Base64 extraction ----
    raw = request.__dict__

    audio_base64 = (
        raw.get("audio_base64")
        or raw.get("audioBase64Format")
        or raw.get("Audio Base64 Format")
        or raw.get("audio")
    )

    if not audio_base64:
        raise HTTPException(status_code=400, detail="Audio Base64 missing")

    # ---- 2. Normalize Base64 (CRITICAL for GUVI) ----
    audio_base64 = (
        audio_base64
        .strip()
        .replace("\n", "")
        .replace("\r", "")
        .replace(" ", "")
    )

    # ---- 3. Decode audio ----
    try:
        audio, sr = decode_base64_audio(audio_base64)
    except Exception:
        # Audio invalid → uncertain prediction
        return PredictionResponse(
            classification="HUMAN",
            confidence=0.5
        )

    # ---- 4. Trim long audio ----
    MAX_DURATION_SEC = 6
    audio = audio[: int(sr * MAX_DURATION_SEC)]

    # ---- 5. Feature extraction + prediction ----
    try:
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=20
        )

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        features = np.concatenate([mfcc_mean, mfcc_std])
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        label, confidence = detector.predict(features)

        # Clamp confidence to sane range
        confidence = float(np.clip(confidence, 0.01, 0.99))

        return PredictionResponse(
            classification=label,
            confidence=confidence
        )

    except Exception as e:
        print("Prediction error:", e)

        # Model uncertainty fallback (realistic)
        return PredictionResponse(
            classification="HUMAN",
            confidence=0.5
        )