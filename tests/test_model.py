import pytest
from model.model import build_model


@pytest.mark.skip("Don't run this in CI")
def test_main_model(data):
    X_tr, X_te = data
    model = build_model()
    model.fit(X_tr)
    predictions = model.predict(X_te)
    revenue_in_2_weeks = predictions.groupby("company_name").sum()
    print(revenue_in_2_weeks)
