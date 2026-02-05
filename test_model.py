import joblib

model = joblib.load("model/detector.pkl")
print("Model loaded:", type(model))
