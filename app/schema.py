from pydantic import BaseModel
from typing import Optional

class AudioRequest(BaseModel):
    # We keep everything optional to avoid 422 errors
    audio_base64: Optional[str] = None
    language: Optional[str] = None
    audio_format: Optional[str] = None

    class Config:
        extra = "allow"   # accept unknown fields safely


class PredictionResponse(BaseModel):
    classification: str
    confidence: float