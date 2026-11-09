import sys

import requests

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696"

booking = {
    "hotel": "City Hotel",
    "lead_time": 180,
    "arrival_date_month": "June",
    "arrival_date_week_number": 24,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "deposit_type": "No Deposit",
    "agent": "9",
    "company": "none",
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 120.0,
}

response = requests.post(f"{url}/predict", json=booking, timeout=30)
response.raise_for_status()
print(response.json())
