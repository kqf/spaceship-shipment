import pytest
import pandas as pd
from model.utils import history_test_split
from model.utils import TimeseriesAnalyzer
from sklearn.metrics import mean_squared_error


@pytest.fixture
def data():
    df = pd.DataFrame(index=pd.date_range("2018-11-01", "2018-12-31"))
    df["revenue"] = 1.
    return df


@pytest.mark.parametrize("n_days", [1, 2, 3, 4, 14])
def test_splits_historical_data(data, n_days):
    train, test = history_test_split(data, n_days=n_days)
    assert len(test) == n_days


def test_makes_prediction(data):
    train, test = history_test_split(data, n_days=14)
    model = TimeseriesAnalyzer(y="revenue").fit(train)
    assert model.score(test) == 0
