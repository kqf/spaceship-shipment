import tqdm
import pandas as pd
import numpy as np

from model.utils import TimeseriesAnalyzer
from model.utils import clean_by_frequency
from model.utils import train_test_split


def read_raw(fname="../data/transactions.data.40181201.csv"):
    raw = pd.read_csv(fname).dropna()

    # Convert to the common units
    #
    raw["shipping_cost"] = raw["shipping_cost"] * raw["exchange_rate"]
    raw["paid"] = raw["paid"] * raw["exchange_rate"]
    raw["revenue"] = raw["paid"] * raw["quantity"] - raw["shipping_cost"]

    raw["timestamp"] = raw["timestamp"].str.replace("401", "201")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"],
                                      infer_datetime_format=True)
    raw["date"] = raw["timestamp"].dt.date
    raw = raw.dropna()  # Remove NaT
    daily = raw[["date", "revenue", "company_name"]].groupby(
        ["date", "company_name"], as_index=False).sum()
    daily.set_index("date", inplace=True)
    return raw.sort_values(by="timestamp")


def read_dataset(fname="data/transactions.data.40181201.csv", n_days=14):
    raw = read_raw(fname)
    daily = raw[["date", "revenue", "company_name"]].groupby(
        ["date", "company_name"], as_index=False).sum()
    daily.set_index("date", inplace=True)
    # Remove eveything that has more less than 16 days of records
    # we want to have a series with at least 2 records
    daily_cleaned = clean_by_frequency(daily, "company_name", n_days + 2)
    return train_test_split(daily_cleaned, "company_name", n_days=n_days)


class GeneralFitter():
    def __init__(self,
                 column="company_name",
                 in_col="revenue",
                 out_col="revenue_predicted"):
        self.models = {}
        self.column = column
        self.in_col = in_col
        self.out_col = out_col

    def fit(self, X):
        self.models = {}
        for company_name, df in tqdm.tqdm(X.groupby(self.column)):
            model = TimeseriesAnalyzer(
                out=self.out_col,
                y=self.in_col,
                daily_seasonality=True)
            self.models[company_name] = model.fit(df)
        return self

    def predict(self, X):
        return pd.concat([
            self._predict(df, m)
            for m, df in tqdm.tqdm(X.groupby(self.column))
        ])

    def _predict(self, df, instance):
        df = pd.DataFrame(df)
        df[self.out_col] = self.models[instance].predict(df)
        return df

    def score(self, X, method="mean"):
        scores = [self.models[m].score(df)
                  for m, df in tqdm.tqdm(X.groupby(self.column))]

        if method == "mean":
            return np.mean(scores)

        return np.sum(scores)


def build_model():
    return GeneralFitter()
