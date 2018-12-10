from model.model import build_model
from model.model import read_dataset


def main():
    model = build_model()
    X_tr, X_te = read_dataset(n_days=14)
    model.fit(X_tr)
    predictions = model.predict(X_te)
    revenue_in_2_weeks = predictions.groupby("company_name").sum()
    print(revenue_in_2_weeks)
    revenue_in_2_weeks.to_csv('predictions.csv', index=False)


if __name__ == "__main__":
    main()
