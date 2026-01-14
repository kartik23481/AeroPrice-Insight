from fastapi import FastAPI
from api.schemas import FlightInput, PredictionResponse
from inference.predict import predict_price

app = FastAPI(
    title="Flight Price Prediction API",
    version="1.0"
)

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_flight_price(payload: FlightInput):
    price = predict_price(payload.dict())
    return {"predicted_price": price}
