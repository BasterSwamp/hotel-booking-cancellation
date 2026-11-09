import pickle
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

THRESHOLD = 0.45

with open(Path(__file__).parent / "model.bin", "rb") as f_in:
    model = pickle.load(f_in)

app = FastAPI(title="Hotel cancellation risk")

Month = Literal[
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


class Booking(BaseModel):
    hotel: Literal["City Hotel", "Resort Hotel"]
    lead_time: int = Field(ge=0)
    arrival_date_month: Month
    arrival_date_week_number: int = Field(ge=1, le=53)
    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)
    adults: int = Field(ge=0)
    children: int = Field(default=0, ge=0)
    babies: int = Field(default=0, ge=0)
    meal: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int = Field(default=0, ge=0, le=1)
    previous_cancellations: int = Field(default=0, ge=0)
    previous_bookings_not_canceled: int = Field(default=0, ge=0)
    reserved_room_type: str
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"]
    agent: str = "none"
    company: str = "none"
    days_in_waiting_list: int = Field(default=0, ge=0)
    customer_type: str
    adr: float = Field(ge=0)


class Prediction(BaseModel):
    cancellation_probability: float
    high_risk: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(booking: Booking) -> Prediction:
    probability = float(model.predict_proba([booking.model_dump()])[0, 1])
    return Prediction(
        cancellation_probability=round(probability, 4),
        high_risk=probability >= THRESHOLD,
    )
