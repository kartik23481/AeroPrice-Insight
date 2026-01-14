from pydantic import BaseModel, Field
import datetime


class FlightInput(BaseModel):
    airline: str
    source: str
    destination: str
    duration: int
    total_stops: int
    additional_info: str
    dep_time_hour: int
    date: datetime.date



class PredictionResponse(BaseModel):
    predicted_price: float


