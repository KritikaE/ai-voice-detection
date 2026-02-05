# app/model.py
import os
import joblib
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "model/detector.pkl")

class VoiceDetector:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, mfcc_features: np.ndarray):
        if mfcc_features.shape != (40,):
            raise ValueError("Expected MFCC shape (40,)")

        X = mfcc_features.reshape(1, -1)

        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        confidence = float(max(proba))
        label = "AI_GENERATED" if pred == 1 else "HUMAN"

        return label, confidence


# Singleton instance (loaded once)
detector = VoiceDetector()