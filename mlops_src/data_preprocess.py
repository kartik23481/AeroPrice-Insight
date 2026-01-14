"""
Data Preprocessing Script
-------------------------

This script:
✔ loads raw flight price data
✔ cleans & normalizes columns
✔ converts duration / stops / dates
✔ splits into train/val/test
✔ logs all steps
✔ saves processed outputs

Usage:
    python mlops_src/data_preprocess.py \
        --input data/raw/flight_price.csv \
        --out_dir data/processed/
"""

import argparse
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# add src to path
SRC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_ROOT, ".."))
sys.path.append(PROJECT_ROOT)

from mlops_src.utils.logger import get_logger


# ========= initialize logging =========
LOG_DIR = os.path.join(PROJECT_ROOT, "mlops_src", "logs")
logger = get_logger("data_preprocess", os.path.join(LOG_DIR, "data_preprocess.log"))
logger.info("===== DATA PREPROCESSING STARTED =====")


# ========= helper functions =========
def change_to_numerical_stops(value: str) -> int:
    v = str(value).strip().lower()
    mapping = {
        "non-stop": 0,
        "non stop": 0,
        "0": 0,
        "1 stop": 1,
        "2 stops": 2,
        "3 stops": 3,
    }
    return mapping.get(v, 4)


def change_duration_type(value: str) -> int:
    v = str(value).strip().lower()
    parts = v.split()
    if len(parts) == 1:
        if "h" in parts[0]:
            return int(parts[0].replace("h", "")) * 60
        return int(parts[0].replace("m", ""))

    hour = int(parts[0].replace("h", ""))
    minutes = int(parts[1].replace("m", ""))
    return hour * 60 + minutes


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # normalize string columns
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].str.lower().str.strip()

    # airline cleanup (exact notebook logic)
    if "airline" in df.columns:
        df["airline"] = (
            df["airline"].str.replace(" premium economy", "")
            .str.replace(" business", "")
            .str.title()
        )

    # total stops numeric
    if "total_stops" in df.columns:
        df["total_stops"] = df["total_stops"].apply(change_to_numerical_stops)

    # duration numeric
    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(change_duration_type)

    # departure time → hour/min
    if "dep_time" in df.columns:
        dep = pd.to_datetime(df["dep_time"], format="%H:%M", errors="coerce")
        df["dep_time_hour"] = dep.dt.hour
        df["dep_time_min"] = dep.dt.minute

    # parse date of journey
    for cand in ["date_of_journey", "date_of_journey_(dd/mm/yyyy)"]:
        if cand in df.columns:
            dtoj = pd.to_datetime(
                df[cand], dayfirst=True, format="%d/%m/%Y", errors="coerce"
            )
            df["dtoj_day"] = dtoj.dt.day
            df["dtoj_month"] = dtoj.dt.month
            df["dtoj_year"] = dtoj.dt.year
            break

    # drop unused
    drop_cols = ["date_of_journey", "dep_time", "route", "arrival_time"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # ensure strings
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str)

    return df.drop_duplicates().reset_index(drop=True)


# ========= main pipeline =========
def run(input_path: str, out_dir: str):
    logger.info(f"📥 Loading raw data from: {input_path}")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_path)
    logger.info(f"Raw shape = {df.shape}")

    df = df.drop_duplicates().reset_index(drop=True)

    df_cleaned = clean_dataframe(df)
    logger.info(f"After cleaning = {df_cleaned.shape}")

    # save cleaned full dataset
    cleaned_path = os.path.join(out_dir, "cleaned_flight_data.csv")
    df_cleaned.to_csv(cleaned_path, index=False)
    logger.info(f"💾 Cleaned data saved → {cleaned_path}")

    # split
    if "price" not in df_cleaned.columns:
        raise ValueError("Column 'price' not found — cannot split target.")

    X = df_cleaned.drop(columns=["price"])
    y = df_cleaned["price"]

    X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_, y_, test_size=0.2, random_state=42)

    train = pd.concat([X_train, y_train], axis=1)
    val = pd.concat([X_val, y_val], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train.to_csv(os.path.join(out_dir, "train_data.csv"), index=False)
    val.to_csv(os.path.join(out_dir, "val_data.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test_data.csv"), index=False)

    logger.info("💾 Saved train/val/test splits")
    logger.info("🚀 Preprocessing Completed Successfully!")


# ========= CLI entry =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flight Price Data Preprocessing")

    parser.add_argument("--input", required=True, help="Path to raw flight CSV")
    parser.add_argument("--out_dir", required=True, help="Directory to save processed files")

    args = parser.parse_args()

    run(args.input, args.out_dir)
