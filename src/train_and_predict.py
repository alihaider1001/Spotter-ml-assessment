"""Train a freight-rate model and generate the required prediction files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_PATH = DATA_DIR / "train-test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
VALIDATION_TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"
DECEMBER_PATH = DATA_DIR / "december-chart-inputs.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "validation_predictions.csv"

TARGET = "posted_rate"
CUTOFF_DATE = pd.Timestamp("2025-10-01")
FEATURE_COLUMNS = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "market_index",
    "quote_signal",
    "month",
    "day_of_week",
    "day_of_month",
]
CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment"]
NUMERIC_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS
]


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw load data and add calendar features.

    The function returns a copy so callers can safely reuse the raw dataframe.
    Missing numeric values are imputed with medians calculated from the input
    dataframe, as required by the assessment specification.
    """
    cleaned = df.copy()

    if "weight" in cleaned.columns:
        cleaned["weight"] = cleaned["weight"].abs()
        cleaned["weight"] = cleaned["weight"].fillna(cleaned["weight"].median())

    if "market_index" in cleaned.columns:
        cleaned["market_index"] = cleaned["market_index"].fillna(
            cleaned["market_index"].median()
        )

    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="raise")
        cleaned["month"] = cleaned["date"].dt.month.astype("int64")
        cleaned["day_of_week"] = cleaned["date"].dt.dayofweek.astype("int64")
        cleaned["day_of_month"] = cleaned["date"].dt.day.astype("int64")

    for column in CATEGORICAL_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype("category")

    return cleaned


def align_inference_features(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    numeric_defaults: dict[str, float],
) -> pd.DataFrame:
    """Add model features absent from a source file and align dtypes.

    The December chart source intentionally contains only its seven original
    columns. For missing model inputs, use training medians; categorical levels
    are aligned to the training categories so XGBoost receives a consistent
    schema at prediction time.
    """
    aligned = frame.copy()

    for column in NUMERIC_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = numeric_defaults[column]
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce").fillna(
            numeric_defaults[column]
        )

    for column in CATEGORICAL_COLUMNS:
        if column not in aligned.columns:
            aligned[column] = reference[column].cat.categories[0]
        aligned[column] = pd.Categorical(
            aligned[column], categories=reference[column].cat.categories
        )

    return aligned[FEATURE_COLUMNS]


def encode_features(
    x_train: pd.DataFrame,
    x_other: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    """Ordinal-encode categoricals using only the training categories."""
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoded_train = x_train.copy()
    encoded_other = x_other.copy()
    encoded_train[CATEGORICAL_COLUMNS] = encoder.fit_transform(
        encoded_train[CATEGORICAL_COLUMNS]
    )
    encoded_other[CATEGORICAL_COLUMNS] = encoder.transform(
        encoded_other[CATEGORICAL_COLUMNS]
    )
    return encoded_train, encoded_other, encoder


def transform_features(
    frame: pd.DataFrame,
    encoder: OrdinalEncoder,
) -> pd.DataFrame:
    """Apply the fitted categorical encoder to another feature frame."""
    transformed = frame.copy()
    transformed[CATEGORICAL_COLUMNS] = encoder.transform(
        transformed[CATEGORICAL_COLUMNS]
    )
    return transformed


def build_model() -> HistGradientBoostingRegressor:
    """Create the tabular regression model used for all predictions."""
    return HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=42,
    )


def main() -> None:
    """Train the model, print holdout metrics, and write both deliverables."""
    train_data = clean_and_engineer(pd.read_csv(TRAIN_PATH))
    train_data["date"] = pd.to_datetime(train_data["date"])

    training_rows = train_data["date"] < CUTOFF_DATE
    internal_validation_rows = ~training_rows
    x_train = train_data.loc[training_rows, FEATURE_COLUMNS]
    y_train = train_data.loc[training_rows, TARGET]
    x_internal_validation = train_data.loc[
        internal_validation_rows, FEATURE_COLUMNS
    ]
    y_internal_validation = train_data.loc[internal_validation_rows, TARGET]
    raw_x_train = x_train
    x_train, x_internal_validation, encoder = encode_features(
        x_train, x_internal_validation
    )

    model = build_model()
    model.fit(
        x_train,
        y_train,
    )

    internal_predictions = model.predict(x_internal_validation)
    internal_mae = mean_absolute_error(y_internal_validation, internal_predictions)
    internal_rmse = np.sqrt(
        mean_squared_error(y_internal_validation, internal_predictions)
    )
    print(f"Internal Validation MAE: {internal_mae:.4f}")
    print(f"Internal Validation RMSE: {internal_rmse:.4f}")

    numeric_defaults: dict[str, float] = {}
    for column in NUMERIC_COLUMNS:
        numeric_defaults[column] = float(pd.to_numeric(raw_x_train[column]).median())

    validation_data = clean_and_engineer(pd.read_csv(VALIDATION_PATH))
    validation_features = align_inference_features(
        validation_data, raw_x_train, numeric_defaults
    )
    validation_features = transform_features(validation_features, encoder)
    validation_predictions = np.clip(model.predict(validation_features), 1.0, None)

    prediction_template = pd.read_csv(VALIDATION_TEMPLATE_PATH)
    prediction_template["predicted_rate"] = validation_predictions
    prediction_template[["load_id", "predicted_rate"]].to_csv(
        PREDICTIONS_PATH, index=False
    )

    december_data = clean_and_engineer(pd.read_csv(DECEMBER_PATH))
    december_features = align_inference_features(
        december_data, raw_x_train, numeric_defaults
    )
    december_features = transform_features(december_features, encoder)
    december_data["predicted_rate"] = np.clip(
        model.predict(december_features), 1.0, None
    )
    december_data.drop(
        columns=["month", "day_of_week", "day_of_month"],
        inplace=True,
        errors="ignore",
    )
    december_data.to_csv(DECEMBER_PATH, index=False)

    print(f"Wrote validation predictions to {PREDICTIONS_PATH}")
    print(f"Updated December predictions at {DECEMBER_PATH}")


if __name__ == "__main__":
    main()
