import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "hotel_bookings.csv"
MODEL_PATH = ROOT / "model.bin"
TEST_START = "2017-04-01"

NUMERICAL = [
    "lead_time",
    "arrival_date_week_number",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "days_in_waiting_list",
    "adr",
]
CATEGORICAL = [
    "hotel",
    "arrival_date_month",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "agent",
    "company",
    "customer_type",
]
FEATURES = NUMERICAL + CATEGORICAL

XGB_PARAMS = {
    "learning_rate": 0.3,
    "max_depth": 4,
    "min_child_weight": 1,
    "n_estimators": 200,
    "random_state": 1,
    "n_jobs": -1,
}


def load_data():
    df = pd.read_csv(DATA_PATH, dtype={"agent": str, "company": str})
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str)
        + "-" + df["arrival_date_month"]
        + "-" + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )

    guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    df = df[(guests > 0) & df["adr"].between(0, 1000)].copy()

    df["children"] = df["children"].fillna(0).astype(int)
    df["agent"] = df["agent"].fillna("none")
    df["company"] = df["company"].fillna("none")
    return df


def records(df):
    return df[FEATURES].to_dict(orient="records")


def main():
    df = load_data()
    df_train = df[df["arrival_date"] < TEST_START]
    df_test = df[df["arrival_date"] >= TEST_START]

    model = make_pipeline(DictVectorizer(), XGBClassifier(**XGB_PARAMS))
    model.fit(records(df_train), df_train["is_canceled"].values)

    test_proba = model.predict_proba(records(df_test))[:, 1]
    print(f"train rows: {len(df_train)}, test rows: {len(df_test)}")
    print(f"test auc: {roc_auc_score(df_test['is_canceled'].values, test_proba):.4f}")

    with open(MODEL_PATH, "wb") as f_out:
        pickle.dump(model, f_out)

    print(f"model saved to {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
