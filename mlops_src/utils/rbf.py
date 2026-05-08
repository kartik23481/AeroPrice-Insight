import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel


class RouteCreator(BaseEstimator, TransformerMixin):
    def __init__(self, route_map):
        self.route_map = route_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # converting numpy → DataFrame 
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=["source", "destination"])

        X2 = X.copy()
        X2["route"] = X2.apply(
            lambda row: self.route_map.get((row["source"], row["destination"]), "Other"),
            axis=1
        )
        return X2[["route"]]
