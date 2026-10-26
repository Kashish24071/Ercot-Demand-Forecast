"""Train, evaluate against the current model, and save only if it is no worse."""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape

from features import FEATURES, make_features

MODEL_PATH = Path("models/model.joblib")
METRICS_PATH = Path("models/metrics.json")
TEST_DAYS = 14
TOLERANCE = 1.02  # the new model may be at most 2% worse (relative) than the current one


def new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=42)


def main() -> None:
    data = make_features(pd.read_parquet("data/demand.parquet")).dropna()
    cutoff = data.index.max() - pd.Timedelta(days=TEST_DAYS)
    train, test = data[data.index <= cutoff], data[data.index > cutoff]
    X_test, y_test = test[FEATURES], test["demand_mw"]

    challenger = new_model().fit(train[FEATURES], train["demand_mw"])
    new_score = mape(y_test, challenger.predict(X_test))
    baseline = mape(y_test, test["lag_168"])  # naive: same hour last week

    old_score = None
    if MODEL_PATH.exists():  # score the current model on the SAME test window
        old_score = mape(y_test, joblib.load(MODEL_PATH).predict(X_test))

    old_text = "none yet" if old_score is None else f"{old_score:.2%}"
    print(f"Train rows: {len(train):,} | test rows: {len(test):,}")
    print(f"Challenger MAPE: {new_score:.2%}")
    print(f"Baseline MAPE:   {baseline:.2%}  (same hour last week)")
    print(f"Current model:   {old_text}")

    if old_score is not None and new_score > old_score * TOLERANCE:
        print("Challenger is worse; keeping the current model.")
        return

    # Save the challenger itself. It never saw the test window, so the next run's
    # comparison is fair (a model refit on all data would have seen the test days).
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(challenger, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "mape": round(new_score, 4),
        "baseline_mape": round(baseline, 4),
        "test_days": TEST_DAYS,
        "train_rows": len(train),
        "data_through": data.index.max().isoformat(),
    }, indent=2))
    print("Saved new model.")


if __name__ == "__main__":
    main()
