import base64
import io
import librosa
from fastapi import HTTPException

TARGET_SR = 16000  # standard sample rate

def decode_base64_audio(audio_base64: str):
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 encoding")

    try:
        audio, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=TARGET_SR,
            mono=True
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Unsupported or corrupted audio file")

    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Empty audio")

    return audio, sr
