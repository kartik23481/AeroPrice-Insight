import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel


class RBFPercentileSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, percentiles=[0.25, 0.5, 0.75], gamma=0.1):
        self.variables = variables
        self.percentiles = percentiles
        self.gamma = gamma

    def fit(self, X, y=None):
        # ---- FIX: convert numpy → DataFrame ----
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # detect variables if not provided
        if not self.variables:
            self.variables = X.select_dtypes(include="number").columns.to_list()

        # save percentiles reference
        self.reference_values_ = {
            col: (
                X.loc[:, col]
                .quantile(self.percentiles)
                .values
                .reshape(-1, 1)
            )
            for col in self.variables
        }
        return self

    def transform(self, X):
        # ---- FIX: convert numpy → DataFrame ----
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.variables)

        outputs = []
        for col in self.variables:
            columns = [f"{col}_rbf_{int(p*100)}" for p in self.percentiles]

            arr = rbf_kernel(
                X[[col]],
                Y=self.reference_values_[col],
                gamma=self.gamma
            )

            out = pd.DataFrame(arr, columns=columns, index=X.index)
            outputs.append(out)

        return pd.concat(outputs, axis=1)



class RouteCreator(BaseEstimator, TransformerMixin):
    def __init__(self, route_map):
        self.route_map = route_map

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # ---- FIX: convert numpy → DataFrame ----
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=["source", "destination"])

        X2 = X.copy()
        X2["route"] = X2.apply(
            lambda row: self.route_map.get((row["source"], row["destination"]), "Other"),
            axis=1
        )
        return X2[["route"]]
