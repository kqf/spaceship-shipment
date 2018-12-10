import datetime as dt

import pandas as pd
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
import logging
from model.externals import SuppressStdoutStderr
logging.getLogger('fbprophet').setLevel(logging.WARNING)


def clean_by_frequency(df, column, frequency):
    counts = df[column].value_counts()
    return df[df[column].isin(counts[counts > frequency].index)]


def history_test_split(df, n_days):
    latest = df.index.max()
    pivot = latest - dt.timedelta(days=n_days)
    return df[df.index <= pivot], df[df.index > pivot]


def train_test_split(df, group_col, n_days):
    train_test = [history_test_split(g, n_days)
                  for _, g in df.groupby(group_col)]
    X_tr, X_te = zip(*train_test)
    return pd.concat(X_tr), pd.concat(X_te)


class TimeseriesAnalyzer(Prophet):
    def __init__(self, y=None, out=None, **kwargs):
        super(TimeseriesAnalyzer, self).__init__(**kwargs)
        self.y = y
        self.out = out or "yhat"
        self.pred_value = None

    def fit(self, X, **kwargs):
        if self.y is not None:
            X = X.rename(columns={self.y: "y"})
        X["ds"] = X.index

        if X.shape[0] < 2:
            self.pred_value = X[self.y][-1]
            return self

        with SuppressStdoutStderr():
            super(TimeseriesAnalyzer, self).fit(X, **kwargs)
        return self

    def predict(self, X, **kwargs):
        output = pd.DataFrame({"ds": X.index})
        output[self.out] = self._predict(output, **kwargs)
        output.set_index(X.index, inplace=True)
        return output[self.out]

    def _predict(self, X, **kwargs):
        if self.pred_value is not None:
            return self.pred_value
        forecast = super(TimeseriesAnalyzer, self).predict(X, **kwargs)
        return forecast["yhat"]

    def score(self, df):
        return mean_squared_error(df[self.y], self.predict(df))
