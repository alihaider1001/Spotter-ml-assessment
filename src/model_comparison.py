"""Empirically compare linear and tree-based freight-rate regressors.

The bake-off uses one fixed temporal holdout and one shared preprocessing path
so the model choice reflects estimator behavior rather than inconsistent data
preparation.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder

from train_and_predict import (
    CATEGORICAL_COLUMNS,
    CUTOFF_DATE,
    FEATURE_COLUMNS,
    TARGET,
    clean_and_engineer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "train-test.csv"


def prepare_features(
    train_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Prepare leak-free model matrices for the historical and future periods.

    ``OrdinalEncoder`` maps unseen locations to ``-1`` instead of raising an
    error, which keeps the comparison representative of future production
    inputs. The holdout uses ``transform`` only: fitting its vocabulary would
    allow future information to influence training.
    """
    # A random split would make nearby market conditions appear in both sets
    # and produce an optimistic estimate. Jan-Sep is training data; October is
    # held out to simulate forecasting the next spot-market period.
    training_rows = train_data["date"] < CUTOFF_DATE
    holdout_rows = ~training_rows

    x_train = train_data.loc[training_rows, FEATURE_COLUMNS].copy()
    y_train = train_data.loc[training_rows, TARGET]
    x_holdout = train_data.loc[holdout_rows, FEATURE_COLUMNS].copy()
    y_holdout = train_data.loc[holdout_rows, TARGET]

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    # Ridge, forests, and histogram boosting require numeric inputs, while the
    # production data contains categorical locations and equipment labels.
    x_train[CATEGORICAL_COLUMNS] = encoder.fit_transform(
        x_train[CATEGORICAL_COLUMNS]
    )
    x_holdout[CATEGORICAL_COLUMNS] = encoder.transform(
        x_holdout[CATEGORICAL_COLUMNS]
    )

    return x_train, y_train, x_holdout, y_holdout


def build_models() -> dict[str, object]:
    """Return the four models used in the empirical comparison.

    Ridge establishes whether rate formation is adequately linear. The
    ensemble models test whether nonlinear distance, seasonality, market, and
    categorical interactions are needed to explain spot-market pricing.
    """
    return {
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            n_jobs=-1,
            random_state=42,
        ),
        "Hist. Gradient Boosting": HistGradientBoostingRegressor(
            random_state=42,
        ),
        "XGBoost": xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            n_jobs=-1,
            random_state=42,
        ),
    }


def print_results(results: list[dict[str, float | str]]) -> None:
    """Print comparison results as a readable ASCII table."""
    headers = ["Model Name", "MAE ($)", "RMSE ($)", "Time (s)"]
    rows = [
        [
            str(result["model"]),
            f"{result['mae']:.2f}",
            f"{result['rmse']:.2f}",
            f"{result['time']:.2f}",
        ]
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    print("\nModel Comparison on October Holdout")
    print(separator)
    header_values = [
        header.ljust(widths[index]) for index, header in enumerate(headers)
    ]
    print("| " + " | ".join(header_values) + " |")
    print(separator)
    for row in rows:
        row_values = [
            value.ljust(widths[index]) for index, value in enumerate(row)
        ]
        print("| " + " | ".join(row_values) + " |")
    print(separator)


def main() -> None:
    """Run the model bake-off and print holdout performance."""
    data = clean_and_engineer(pd.read_csv(TRAIN_PATH))
    data["date"] = pd.to_datetime(data["date"])
    x_train, y_train, x_holdout, y_holdout = prepare_features(data)

    results: list[dict[str, float | str]] = []
    for model_name, model in build_models().items():
        # Timing fit plus inference captures the operational cost of selecting
        # a model, not merely its offline accuracy on the holdout.
        start_time = time.perf_counter()
        model.fit(x_train, y_train)
        predictions = model.predict(x_holdout)
        elapsed_seconds = time.perf_counter() - start_time

        results.append(
            {
                "model": model_name,
                "mae": mean_absolute_error(y_holdout, predictions),
                "rmse": np.sqrt(mean_squared_error(y_holdout, predictions)),
                "time": elapsed_seconds,
            }
        )

    print_results(results)


if __name__ == "__main__":
    main()
