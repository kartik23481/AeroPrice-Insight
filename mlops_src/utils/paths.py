# mlops_src/utils/paths.py

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw", "flight_price.csv")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

TRAIN_DATA = os.path.join(DATA_PROCESSED_DIR, "train_data.csv")
VAL_DATA = os.path.join(DATA_PROCESSED_DIR, "val_data.csv")
TEST_DATA = os.path.join(DATA_PROCESSED_DIR, "test_data.csv")

BACKEND_ARTIFACTS = os.path.join(PROJECT_ROOT, "backend_artifacts")

TRANSFORMER_PATH = os.path.join(BACKEND_ARTIFACTS, "latest_column_transformer.joblib")
MODEL_PATH = os.path.join(BACKEND_ARTIFACTS, "xgb_flight_price_model.joblib")

DOCKER_BACKEND_ARTIFACTS = os.path.join(
    PROJECT_ROOT, "docker_backend", "artifacts"
)
