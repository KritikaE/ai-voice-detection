from pydantic import BaseModel, Field
from typing import Optional


class AudioRequest(BaseModel):
    audio_base64: Optional[str] = Field(
        None,
        alias="audioBase64Format"
    )
    language: Optional[str] = None
    audio_format: Optional[str] = None

    class Config:
        allow_population_by_field_name = True
        extra = "allow"


class PredictionResponse(BaseModel):
    classification: str
    confidence: float
