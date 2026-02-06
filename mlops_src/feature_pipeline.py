import os
import sys
import joblib
import pandas as pd

# ensure imports from project root work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from utils.feature_utils import (
    is_same_region,
    part_of_month,
    part_of_day,
    make_month_object,
    duration_category,
    direct_flight,
    ToDataFrame,
)


from utils.rbf import RouteCreator, RBFPercentileSimilarity

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    FunctionTransformer,
    PowerTransformer
)
from sklearn.decomposition import PCA
from feature_engine.encoding import RareLabelEncoder, MeanEncoder
from feature_engine.datetime import DatetimeFeatures
from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer

from logging import getLogger, basicConfig

# ==============================
# Logging Setup
# ==============================
from mlops_src.utils.logger import get_logger

LOG_DIR = os.path.join(ROOT, "mlops_src", "logs")
logger = get_logger("feature_pipeline", os.path.join(LOG_DIR, "feature_pipeline.log"))
logger.info("===== FEATURE PIPELINE STARTED =====")


def build_pipeline():
    """
    Build ColumnTransformer exactly matching updated feature_engineering.ipynb
    """

    logger.info("Building airline pipeline...")
    airline_transformer = Pipeline(steps=[
        ("grouper", RareLabelEncoder(tol=0.1, n_categories=2, replace_with="Other")),
        ("onehotencoding", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])

    logger.info("Building source-destination pipeline...")
    route_map = {
        ("delhi", "cochin"): "1",
        ("kolkata", "banglore"): "2",
        ("mumbai", "hyderabad"): "3",
        ("banglore", "new delhi"): "4",
        ("banglore", "delhi"): "5",
        ("chennai", "kolkata"): "6"
    }

    sor_des_trans = Pipeline(steps=[
        ("create_route", RouteCreator(route_map=route_map)),
        ("onehotencoding", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])

    source_destination_trans = FeatureUnion(transformer_list=[
        ("part1", sor_des_trans),
        ("part2", FunctionTransformer(func=is_same_region))
    ])

    logger.info("Building departure hour pipeline...")
    dep_time_hour_union = Pipeline(steps=[
        ("part1", FunctionTransformer(func=part_of_day)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    logger.info("Building month pipeline...")
    month_pipeline = Pipeline(steps=[
        ("make_month_object", FunctionTransformer(func=make_month_object)),
        ("onehotencoding", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])

    logger.info("Building dtoj_day pipeline...")
    dtoj_day_pipeline = Pipeline(steps=[
        ("bucket", FunctionTransformer(func=part_of_month)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    logger.info("Building weekend pipeline...")
    weekend_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    logger.info("Building duration pipelines...")
    duration_cat_pipeline = Pipeline([
        ("cat", FunctionTransformer(func=duration_category)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    duration_num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    duration_union = FeatureUnion([
        ("numeric", duration_num_pipeline),
        ("categorical", duration_cat_pipeline)
    ])

    logger.info("Building total_stops pipeline...")
    total_stops_pipeline = Pipeline(steps=[
        ("stops_num", SimpleImputer(strategy="most_frequent")),
        ("direct_flag", FunctionTransformer(func=direct_flight))
    ])

    logger.info("Building additional_info pipeline...")
    info_pipe1 = Pipeline(steps=[
        ("to_df", ToDataFrame(["additional_info"])),
        ("group", RareLabelEncoder(tol=0.2, n_categories=2, replace_with="Other")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    info_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("union", info_pipe1)
    ])

    logger.info("Packing into ColumnTransformer...")

    column_transformer = ColumnTransformer(
        transformers=[
            ("tf1", airline_transformer, ["airline"]),
            ("tf2", source_destination_trans, ["source", "destination"]),
            ("tf3", dep_time_hour_union, ["dep_time_hour"]),
            ("tf4", dtoj_day_pipeline, ["dtoj_day"]),
            ("tf5", month_pipeline, ["dtoj_month"]),
            ("tf6", weekend_pipeline, ["is_weekend"]),
            ("tf7", duration_union, ["duration"]),
            ("tf8", total_stops_pipeline, ["total_stops"]),
            ("tf9", info_transformer, ["additional_info"])
        ],
        remainder="drop"
    )

    return column_transformer


# def build_pipeline():
#     """
#     Build ColumnTransformer exactly as notebook.
#     """

#     logger.info("Building airline pipeline...")
#     airline_transformer = Pipeline(steps=[
#         ("grouper", RareLabelEncoder(tol=0.1, n_categories=2, replace_with="Other")),
#         ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
#     ])

#     logger.info("Route encoding...")
#     route_map = {
#         ("delhi", "cochin"): "1",
#         ("kolkata", "banglore"): "2",
#         ("mumbai", "hyderabad"): "3",
#         ("bangalore", "newdelhi"): "4",
#         ("bangalore", "delhi"): "5",
#         ("chennai", "kolkata"): "6",
#     }

#     sor_des_trans = Pipeline(steps=[
#         ("create_route", RouteCreator(route_map=route_map)),
#         ("onehotencoding",OneHotEncoder(sparse_output=False,handle_unknown='ignore'))
#     ])

#     source_destination_union = FeatureUnion(transformer_list=[
#         ("part1",sor_des_trans),
#         ("part2",FunctionTransformer(func=is_same_region))
#     ])

#     logger.info("Dep time hour union...")
#     dep_time_hour_union = FeatureUnion(transformer_list=[
#         ("daypart", FunctionTransformer(func=part_of_day)),
#         ("scale", MinMaxScaler()),
#     ])

#     pipe3 = Pipeline(steps=[
#         ("month_obj", FunctionTransformer(func=make_month_object)),
#         ("mean_encode", MeanEncoder()),
#         ("scale", StandardScaler()),
#     ])

#     dtoj_month_transformer = Pipeline(steps=[
#         ("scale", MinMaxScaler()),
#         ("pca", PCA(n_components=1)),
#     ])

#     dtoj_month_union = FeatureUnion(transformer_list=[
#         ("pca_path", dtoj_month_transformer),
#         ("month_path", pipe3),
#     ])

#     pipe = Pipeline(steps=[
#         ("month_bucket", FunctionTransformer(func=find_part_of_month)),
#         ("mean_encode", MeanEncoder()),
#         ("scale", StandardScaler()),
#     ])

#     dtoj_day_union = FeatureUnion(transformer_list=[
#         ("mean_part", pipe),
#         ("scaled", MinMaxScaler()),
#     ])

#     logger.info("Duration + stops pipelines...")
#     duration_pipe1 = Pipeline(steps=[
#         ("rbf", RBFPercentileSimilarity()),
#         ("scaler", PowerTransformer()),
#     ])

#     duration_pipe2 = Pipeline(steps=[
#         ("category", FunctionTransformer(func=duration_category)),
#         ("encode", OrdinalEncoder(categories=[["short", "medium", "long"]])),
#     ])

#     duration_union = FeatureUnion(transformer_list=[
#         ("rbf_path", duration_pipe1),
#         ("duration_cat", duration_pipe2),
#         ("scaled", StandardScaler()),
#     ])

#     duration_transformer = Pipeline(steps=[
#         ("winsor", Winsorizer(capping_method="iqr", fold=1.5)),
#         ("impute", SimpleImputer(strategy="median")),
#         ("union", duration_union),
#     ])

#     numeric_transformer = Pipeline(steps=[
#         ("scale", StandardScaler()),
#         ("pca", PCA(n_components=1)),
#     ])

#     pipe4 = Pipeline(steps=[
#         ("remove", FunctionTransformer(func=remove_duration)),
#     ])

#     total_stops_union = FeatureUnion(transformer_list=[
#         ("scaled", numeric_transformer),
#         ("remove_path", pipe4),
#     ])

#     info_pipe1 = Pipeline(steps=[
#         ("to_df", ToDataFrame(["additional_info"])),
#         ("rare", RareLabelEncoder(tol=0.1, n_categories=2, replace_with="Other")),
#         ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
#     ])

#     info_transformer = Pipeline(steps=[
#         ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
#         ("union", info_pipe1),
#     ])

#     logger.info("Packing into ColumnTransformer...")

#     column_transformer = ColumnTransformer(
#         transformers=[
#             ("airline", airline_transformer, ["airline"]),
#             ("src_dst", source_destination_union, ["source", "destination"]),
#             ("dtoj_day", dtoj_day_union, ["dtoj_day"]),
#             ("dep_time", dep_time_hour_union, ["dep_time_hour"]),
#             ("dtoj_month", dtoj_month_union, ["dtoj_month", "is_weekend"]),
#             ("total_stops", total_stops_union, ["duration", "total_stops"]),
#             ("duration", duration_transformer, ["duration"]),
#             ("info", info_transformer, ["additional_info"]),
#         ],
#         remainder="drop",
#     )

#     return column_transformer



def fit_and_save(train_df: pd.DataFrame, y: pd.Series, artifacts_dir: str = None):
    """
    Build, fit, and save column transformer INTO backend_artifacts folder.
    """

    if artifacts_dir is None:
        artifacts_dir = os.path.join(ROOT, "backend_artifacts")

    logger.info(f"Artifacts dir resolved → {artifacts_dir}")
    os.makedirs(artifacts_dir, exist_ok=True)

    # create weekend + drop redundant
    date = pd.to_datetime(train_df.rename(columns={
        "dtoj_year": "year",
        "dtoj_month": "month",
        "dtoj_day": "day",
    })[["year", "month", "day"]])

    train_df = train_df.assign(is_weekend=(date.dt.weekday >= 5).astype(int))
    train_df = train_df.drop(columns=["dep_time_min", "dtoj_year"], errors="ignore")

    logger.info("Building pipeline...")
    column_transformer = build_pipeline()

    logger.info("Fitting pipeline on training data...")
    column_transformer.fit(train_df, y)

    save_path = os.path.join(artifacts_dir, "latest_column_transformer.joblib")
    joblib.dump(column_transformer, save_path)

    logger.info(f"Transformer saved → {save_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to train_data.csv")
    parser.add_argument("--artifacts", required=False, help="Artifacts directory")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    y = df["price"]
    X = df.drop(columns=["price"])

    fit_and_save(X, y, args.artifacts)