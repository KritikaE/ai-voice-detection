from fastapi import FastAPI, Header, HTTPException, Depends
from app.schema import AudioRequest, PredictionResponse
from app.audio import decode_base64_audio
from app.model import detector
import librosa
import numpy as np
import os

API_KEY = os.getenv("API_KEY", "voice-detection-secret-key")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

app = FastAPI(title="AI Voice Detection API")

@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: AudioRequest,
    _: str = Depends(verify_api_key)
):
    try:
        audio, sr = decode_base64_audio(request.audio_base64)

        # MFCC extraction (contract: 40)
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )
        mfcc_mean = np.mean(mfcc, axis=1)

        classification, confidence = detector.predict(mfcc_mean)

        return PredictionResponse(
            classification=classification,
            confidence=confidence
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
