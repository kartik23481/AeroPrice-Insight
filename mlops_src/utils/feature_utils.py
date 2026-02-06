# feature_utils.py

import numpy as np
import pandas as pd

north_cities = ['delhi', 'new delhi']

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class ToDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.columns)

    
north_cities = ['delhi', 'new delhi']

def is_same_region(df):
    if isinstance(df, np.ndarray): # Ensure input is DataFrame
        df = pd.DataFrame(df, columns=['source', 'destination'])

    return (
        df.assign(
            same_region=lambda df_: np.select(
                [df_.source.isin(north_cities) & df_.destination.isin(north_cities)],
                [1],
                default=[0]
            )
        )
        .drop(columns=['source', 'destination'])
    )


def is_north(temp):
    # Ensure input is DataFrame
    if isinstance(temp, np.ndarray):
        temp = pd.DataFrame(temp, columns=['source', 'destination'])
    
    return (
        temp.assign(
            source_is_north=lambda df_: np.select([df_.source.isin(north_cities)], [1], default=[0]),
            destination_is_north=lambda df_: np.select([df_.destination.isin(north_cities)], [1], default=[0])
        )
        .drop(columns=['source', 'destination'])
    )

def part_of_month(df):
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df, columns=["dtoj_day"])

    return (
        df.assign(
            part_of_month=np.select(
                [
                    df.dtoj_day.between(1, 10),
                    df.dtoj_day.between(11, 20),
                    df.dtoj_day.between(21, 31),
                ],
                ["early", "mid", "late"],
                default="unknown",
            )
        )
        .drop(columns=["dtoj_day"])
    )



def part_of_day(df_):

    if isinstance(df_, np.ndarray):
        df_ = pd.DataFrame(df_, columns=["dep_time_hour"])

    return (
        df_.assign(
            part_of_day=np.select(
                [
                    df_.dep_time_hour.between(4, 12, inclusive="left"),
                    df_.dep_time_hour.between(12, 16, inclusive="left"),
                    df_.dep_time_hour.between(16, 20, inclusive="left"),
                ],
                ["morning", "afternoon", "evening"],
                default="night",
            )
        )
        .drop(columns=["dep_time_hour"])
    )



def make_month_object(df):
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df, columns=['dtoj_month'])
    
    return (
        df.assign(
            dtoj_month=df.dtoj_month.astype('object')
        )
    )

def direct_flight(df):
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df, columns=['total_stops'])
    
    return (
        df.assign(is_direct_flight=df.total_stops.eq(0).astype(int))
    )

def have_info(X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=['additional_info'])
    
    return X.assign(additional_info=X.additional_info.ne("No Info").astype(int))

def duration_category(X, short=180, med=400):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=['duration'])
    
    return (
        X.assign(
            duration_cat=np.select(
                [X.duration.between(0, short, inclusive="left"),
                 X.duration.between(short, med, inclusive="left")],
                ["short", "medium"],
                default="long"
            )
        )
        .drop(columns="duration")
    )
