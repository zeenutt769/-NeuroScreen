from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib, json
import numpy as np

app = FastAPI(title="ASD Detection API 🧠", version="1.0")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load models
models = {
    "random_forest_adult":    joblib.load("models/rf_adult.pkl"),
    "random_forest_child":    joblib.load("models/rf_child.pkl"),
    "xgboost_adult":          joblib.load("models/xgb_adult.pkl"),
    "xgboost_child":          joblib.load("models/xgb_child.pkl"),
    "adaboost_adult":         joblib.load("models/ada_adult.pkl"),
    "adaboost_child":         joblib.load("models/ada_child.pkl"),
    "decision_tree_adult":    joblib.load("models/cart_adult.pkl"),
    "decision_tree_child":    joblib.load("models/cart_child.pkl"),
    "gradient_boosting_adult":joblib.load("models/gb_adult.pkl"),
    "gradient_boosting_child":joblib.load("models/gb_child.pkl"),
}
scalers = {
    "adult": joblib.load("models/scaler_adult.pkl"),
    "child": joblib.load("models/scaler_child.pkl"),
}
with open("models/feature_names.json") as f:
    FEATURE_NAMES = json.load(f)

class PredictRequest(BaseModel):
    features: list[float]          # 18 values
    model_name: str = "xgboost_adult"

@app.get("/api/info")
def root():
    return {
        "message": "ASD Detection API is live 🧠",
        "available_models": list(models.keys()),
        "expected_features": FEATURE_NAMES
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if req.model_name not in models:
        return {"error": f"Model '{req.model_name}' not found. Choose from: {list(models.keys())}"}

    dataset = "child" if "child" in req.model_name else "adult"
    scaler  = scalers[dataset]

    X = np.array(req.features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    model      = models[req.model_name]
    prediction = model.predict(X_scaled)[0]
    proba      = model.predict_proba(X_scaled)[0].tolist()

    return {
        "model_used":        req.model_name,
        "prediction":        int(prediction),
        "prediction_label":  "ASD Positive ✅" if prediction == 1 else "ASD Negative ❌",
        "confidence":        round(max(proba) * 100, 2),
        "probabilities":     {"ASD_negative": round(proba[0], 4), "ASD_positive": round(proba[1], 4)}
    }

@app.get("/features")
def get_features():
    return {"feature_names": FEATURE_NAMES, "total": len(FEATURE_NAMES)}

# Mount the frontend directory to serve the frontend UI natively
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
