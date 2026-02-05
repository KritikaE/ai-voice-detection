from pydantic import BaseModel
from typing import Optional

class AudioRequest(BaseModel):
    audio_base64: str
    language: Optional[str] = None
    audio_format: Optional[str] = None


class PredictionResponse(BaseModel):
    classification: str
    confidence: float