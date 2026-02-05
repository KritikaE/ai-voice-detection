# schemas.py
from pydantic import BaseModel, Field

class AudioRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded MP3 audio")

class PredictionResponse(BaseModel):
    classification: str = Field(..., example="HUMAN")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.85)
