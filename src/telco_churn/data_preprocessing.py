from __future__ import annotations

import argparse

from sklearn.model_selection import train_test_split

from .feature_engineering import clean_telco_data, preprocess_features
from .utils import (
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    ensure_project_dirs,
    load_raw_data,
    save_json,
    save_model,
)


def build_processed_dataset(raw_path=RAW_DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    ensure_project_dirs()

    df = clean_telco_data(load_raw_data(raw_path))
    X = df.drop(columns=["Churn"])

    y = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .map({
            "Yes": 1,
            "No": 0,
            "1": 1,
            "0": 0,
        })
    )

    if y.isna().any():
        raise ValueError("Churn column contains unexpected values.")

    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train, scaler = preprocess_features(X_train, fit_scaler=True)
    feature_columns = list(X_train.columns)
    X_test, _ = preprocess_features(X_test, scaler=scaler, feature_columns=feature_columns)

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    save_model(scaler, MODELS_DIR / "scaler.pkl")
    save_json(feature_columns, MODELS_DIR / "feature_columns.json")

    return X_train, X_test, y_train, y_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and preprocess the Telco churn dataset.")
    parser.add_argument("--raw-path", default=RAW_DATA_PATH, help="Path to the raw Telco CSV file.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = build_processed_dataset(
        raw_path=args.raw_path,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"Saved processed data: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Train churn rate={y_train.mean():.3f}, test churn rate={y_test.mean():.3f}")


if __name__ == "__main__":
    main()
