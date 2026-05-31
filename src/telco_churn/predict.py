from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .feature_engineering import clean_telco_data, preprocess_features
from .utils import MODELS_DIR, get_feature_columns, load_model


def assign_risk_segment(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"


def score_customers(
    customers: pd.DataFrame,
    model=None,
    scaler=None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    if model is None:
        model = load_model(MODELS_DIR / "final_model.pkl")
    if scaler is None:
        scaler = load_model(MODELS_DIR / "scaler.pkl")
    if feature_columns is None:
        feature_columns = get_feature_columns()

    identifiers = pd.DataFrame(index=customers.index)
    if "customerID" in customers.columns:
        identifiers["customerID"] = customers["customerID"]

    features_source = customers.drop(columns=["Churn"], errors="ignore")
    features_source = clean_telco_data(features_source, drop_customer_id=True)
    features, _ = preprocess_features(features_source, scaler=scaler, feature_columns=feature_columns)

    probabilities = model.predict_proba(features)[:, 1]
    output = identifiers.loc[features.index].copy()
    output["churn_probability"] = probabilities
    output["risk_segment"] = output["churn_probability"].map(assign_risk_segment)
    return output.sort_values("churn_probability", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score customers with the saved churn model.")
    parser.add_argument("--input", required=True, help="Path to a customer CSV file.")
    parser.add_argument("--output", default="reports/customer_scores.csv", help="Where to save scores.")
    args = parser.parse_args()

    scores = score_customers(pd.read_csv(args.input))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path, index=False)
    print(f"Saved {len(scores)} scored customers to {output_path}")


if __name__ == "__main__":
    main()
