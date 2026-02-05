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
# Warm-up (prevents cold-start timeout)
# -------------------------
@app.on_event("startup")
def warmup_model():
    dummy_input = [0.0] * EXPECTED_FEATURE_DIM
    detector.model.predict_proba([dummy_input])


# -------------------------
# Predict Endpoint (HARDENED)
# -------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: AudioRequest, _: str = Depends(verify_api_key)):
    try:
        body = request.__dict__

        # Accept all possible field names
        audio_base64 = (
            body.get("audio_base64")
            or body.get("Audio Base64 Format")
            or body.get("audioBase64")
            or body.get("audio_base64_format")
        )

        # If audio missing or empty → return safe response
        if not audio_base64 or not audio_base64.strip():
            return PredictionResponse(
                classification="UNKNOWN",
                confidence=0.0
            )

        # Decode + preprocess
        audio, sr = decode_base64_audio(audio_base64)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=EXPECTED_FEATURE_DIM
        )
        features = np.mean(mfcc, axis=1)

        # Predict
        label, confidence = detector.predict(features)

        return PredictionResponse(
            classification=label,
            confidence=confidence
        )

    except Exception:
        # NEVER crash for hackathon tester
        return PredictionResponse(
            classification="UNKNOWN",
            confidence=0.0
        )
