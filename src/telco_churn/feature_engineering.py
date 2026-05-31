from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.preprocessing import StandardScaler


SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "CustomerValue",
    "AvgCharges",
    "ChargeDifference",
]

EXPECTED_CATEGORICAL_VALUES = {
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}


def clean_telco_data(df: pd.DataFrame, drop_customer_id: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    if drop_customer_id and "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "Churn" in df.columns and df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def _validate_no_missing(df: pd.DataFrame, columns: Iterable[str], step_name: str) -> None:
    missing_cols = [col for col in columns if col in df.columns and df[col].isna().any()]
    if missing_cols:
        raise ValueError(f"Missing or unknown values after {step_name}: {missing_cols}")


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["AvgCharges"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ChargeDifference"] = df["MonthlyCharges"] - df["AvgCharges"]

    service_mapping = {"Yes": 1, "No": 0, "No internet service": 0}
    for col in SERVICE_COLS:
        df[col] = df[col].map(service_mapping)
    _validate_no_missing(df, SERVICE_COLS, "service mapping")

    df["ServiceCoverageRatio"] = df[SERVICE_COLS].sum(axis=1) / len(SERVICE_COLS)
    df["CustomerValue"] = df["MonthlyCharges"] * df["tenure"]

    contract_mapping = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["ContractRisk"] = df["Contract"].map(contract_mapping)
    _validate_no_missing(df, ["ContractRisk"], "contract mapping")
    df = df.drop(columns=["Contract"])

    df["EarlyLifecycle"] = (df["tenure"] < 12).astype(int)
    df["SupportGap"] = ((df["TechSupport"] == 0) & (df["OnlineSecurity"] == 0)).astype(int)

    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {"Yes": 1, "No": 0, "Female": 1, "Male": 0}

    for col in BINARY_COLS:
        df[col] = df[col].map(mapping)
    _validate_no_missing(df, BINARY_COLS, "binary mapping")

    return df


def encode_nominal_columns(df: pd.DataFrame, inference_mode: bool) -> pd.DataFrame:
    df = df.copy()

    for col, values in EXPECTED_CATEGORICAL_VALUES.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=values)

    # During inference, keep all category indicator columns and then align to the
    # trained feature set. This preserves a selected category even for one-row
    # predictions, where drop_first=True would otherwise remove the only value.
    return pd.get_dummies(df, drop_first=not inference_mode)


def preprocess_features(
    X: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
    feature_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    inference_mode = feature_columns is not None

    X = add_business_features(X)
    X = encode_binary_columns(X)
    X = encode_nominal_columns(X, inference_mode=inference_mode)

    if feature_columns is not None:
        X = X.reindex(columns=list(feature_columns), fill_value=0)

    if scaler is None:
        if not fit_scaler:
            raise ValueError("A fitted scaler is required when fit_scaler=False.")
        scaler = StandardScaler()

    cols_to_scale = [col for col in NUMERIC_COLS if col in X.columns]
    if fit_scaler:
        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    else:
        X[cols_to_scale] = scaler.transform(X[cols_to_scale])

    return X, scaler
