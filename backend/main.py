# backend/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# ---------- 1. Define input schema (must match features used in training) ----------

class AirbnbInput(BaseModel):
    neighbourhood_group: str
    neighbourhood: str
    room_type: str
    latitude: float
    longitude: float
    minimum_nights: int
    number_of_reviews: int
    reviews_per_month: float
    calculated_host_listings_count: int
    availability_365: int


# ---------- 2. Create FastAPI app (this MUST be named `app` and NOT indented) ----------

app = FastAPI(
    title="Airbnb NYC Price Prediction API",
    version="1.0.0",
    description="Predict nightly price for NYC Airbnb listings using a trained ML model.",
)

# ---------- 3. Load the trained model ----------

# price_model.pkl is saved in ../notebook/ relative to this file
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "notebook", "price_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Could not find model file at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


# ---------- 4. Simple health check endpoint ----------

@app.get("/")
def read_root():
    return {"message": "Airbnb NYC Price API is running ✅"}


# ---------- 5. Prediction endpoint ----------

@app.post("/predict")
def predict_price(payload: AirbnbInput):
    """
    Take Airbnb listing features and return predicted nightly price.
    """
    # Convert Pydantic model → DataFrame with one row
    input_df = pd.DataFrame([payload.dict()])

    # Predict using the trained pipeline
    pred = model.predict(input_df)[0]

    return {
        "predicted_price": round(float(pred), 2),
        "currency": "USD",
    }
































































