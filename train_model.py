
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = Path("data/used_cars.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "car_price_model.joblib"
INFO_FILE = MODEL_DIR / "model_info.json"


def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE.resolve()}"
        )

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    if df.empty:
        raise ValueError("The dataset is empty.")

    # Find the target column using common names.
    target_candidates = {
        "sellingprice", "price", "selling_price",
        "selling_price_inr", "resaleprice", "carprice",
        "predictedprice"
    }

    target_col = next(
        (
            col for col in df.columns
            if normalize_name(col) in target_candidates
        ),
        None
    )

    if target_col is None:
        raise ValueError(
            "Could not identify the selling-price column.\n"
            f"Available columns: {df.columns.tolist()}\n"
            "Edit target_candidates in train_model.py to match your CSV."
        )

    # Remove completely empty columns and common CSV index columns.
    df = df.dropna(axis=1, how="all")
    df = df.loc[
        :, ~df.columns.str.match(r"^Unnamed:", case=False)
    ]

    # Convert the target to numbers, handling commas and currency symbols.
    target = pd.to_numeric(
        df[target_col].astype(str).str.replace(
            r"[^0-9.\-]", "", regex=True
        ),
        errors="coerce"
    )

    df = df.loc[target.notna()].copy()
    df[target_col] = target.loc[target.notna()]

    # A car price must be positive.
    df = df[df[target_col] > 0].copy()

    if len(df) < 10:
        raise ValueError(
            "Not enough valid rows to train reliably. "
            "Check the price column and dataset."
        )

    # Drop rows without a usable target, then prepare the features.
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Remove identifier-like columns and constant columns.
    drop_cols = []
    for col in X.columns:
        name = normalize_name(col)
        if name in {"id", "carid", "index", "listingid"}:
            drop_cols.append(col)
        elif X[col].nunique(dropna=True) <= 1:
            drop_cols.append(col)

    X = X.drop(columns=drop_cols)

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")

    # Convert numeric-looking text columns to numeric values where
    # at least 90% of their non-empty values can be parsed.
    for col in X.columns:
        if X[col].dtype == "object":
            cleaned = pd.to_numeric(
                X[col].astype(str).str.replace(
                    r"[,₹$]", "", regex=True
                ),
                errors="coerce"
            )
            non_empty = X[col].notna().sum()
            if non_empty and cleaned.notna().sum() / non_empty >= 0.90:
                X[col] = cleaned

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [
        col for col in X.columns if col not in numeric_cols
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []

    if numeric_cols:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print("Training Linear Regression model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    # Save input information for the Streamlit application.
    feature_info = {}

    for col in X.columns:
        series = X[col]

        if col in numeric_cols:
            valid = series.dropna()
            default = float(valid.median()) if not valid.empty else 0.0
            feature_info[col] = {
                "type": "numeric",
                "default": default,
                "min": float(valid.min()) if not valid.empty else 0.0,
                "max": float(valid.max()) if not valid.empty else 100.0
            }
        else:
            values = (
                series.dropna().astype(str).value_counts().index.tolist()
            )
            feature_info[col] = {
                "type": "categorical",
                "default": values[0] if values else "Unknown",
                "options": values[:50],
                "unique_count": int(series.nunique(dropna=True))
            }

    info = {
        "target_col": target_col,
        "features": X.columns.tolist(),
        "feature_info": feature_info,
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        },
        "rows_used": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dropped_columns": drop_cols
    }

    with open(INFO_FILE, "w", encoding="utf-8") as file:
        json.dump(info, file, indent=4)

    # Save test predictions for charts in the application.
    pd.DataFrame({
        "Actual Price": y_test.to_numpy(),
        "Predicted Price": predictions
    }).to_csv(MODEL_DIR / "test_predictions.csv", index=False)

    print("\n========== MODEL RESULTS ==========")
    print("Target column:", target_col)
    print("Features:", X.columns.tolist())
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R² score: {r2:.4f}")
    print("\nModel saved to:", MODEL_FILE)
    print("Model information saved to:", INFO_FILE)


if __name__ == "__main__":
    main()
