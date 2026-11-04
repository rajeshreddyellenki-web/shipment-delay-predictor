"""
Shipment Delay Predictor
------------------------
Trains a machine-learning model that predicts whether a shipment will
arrive late, and shows which factors drive delays.

Usage:
    python shipment_delay_predictor.py --generate-sample   # create demo shipments.csv
    python shipment_delay_predictor.py                     # train on shipments.csv
    python shipment_delay_predictor.py --predict           # score an example shipment

Input file (shipments.csv), one row per shipment:
    carrier, shipping_mode, origin_region, destination_region, distance_km,
    weight_kg, priority, order_month, day_of_week, weather_severity,
    traffic_index, is_holiday_week, carrier_past_delay_rate, is_delayed (0/1)

Outputs:
    delay_model.joblib        : trained model pipeline
    model_report.png          : confusion matrix + feature importance
    (console)                 : accuracy, precision, recall, F1, ROC-AUC
"""

import argparse
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt

DATA_FILE = "shipments.csv"
MODEL_FILE = "delay_model.joblib"
REPORT_FILE = "model_report.png"
TARGET = "is_delayed"

CATEGORICAL = ["carrier", "shipping_mode", "origin_region", "destination_region", "priority"]
NUMERIC = [
    "distance_km", "weight_kg", "order_month", "day_of_week",
    "weather_severity", "traffic_index", "is_holiday_week", "carrier_past_delay_rate",
]


# ---------------------------------------------------------------------------
# 1. Sample data generator
# ---------------------------------------------------------------------------
def generate_sample_data(n: int = 6000, seed: int = 42) -> None:
    """Create a realistic demo shipments.csv where delays depend on known factors."""
    rng = np.random.default_rng(seed)

    carriers = {"FastFreight": 0.10, "GlobalLogix": 0.18, "SwiftMove": 0.25, "BudgetShip": 0.35}
    modes = {"Air": -1.0, "Road": 0.0, "Rail": 0.3, "Sea": 1.0}  # delay tendency
    regions = ["North", "South", "East", "West", "Central"]

    carrier = rng.choice(list(carriers), n)
    mode = rng.choice(list(modes), n, p=[0.15, 0.50, 0.15, 0.20])
    df = pd.DataFrame(
        {
            "carrier": carrier,
            "shipping_mode": mode,
            "origin_region": rng.choice(regions, n),
            "destination_region": rng.choice(regions, n),
            "distance_km": rng.integers(50, 8000, n),
            "weight_kg": np.round(rng.gamma(2.0, 120, n), 1),
            "priority": rng.choice(["Standard", "Express"], n, p=[0.75, 0.25]),
            "order_month": rng.integers(1, 13, n),
            "day_of_week": rng.integers(0, 7, n),
            "weather_severity": np.round(rng.uniform(0, 10, n), 1),
            "traffic_index": np.round(rng.uniform(0, 10, n), 1),
            "is_holiday_week": rng.choice([0, 1], n, p=[0.88, 0.12]),
        }
    )
    df["carrier_past_delay_rate"] = [
        round(float(np.clip(carriers[c] + rng.normal(0, 0.03), 0, 1)), 3) for c in carrier
    ]

    # Delay probability: logistic function of the real drivers + noise
    score = (
        -5.4
        + 0.00025 * df["distance_km"]
        + 0.0007 * df["weight_kg"]
        + 0.22 * df["weather_severity"]
        + 0.18 * df["traffic_index"]
        + 1.1 * df["is_holiday_week"]
        + 4.0 * df["carrier_past_delay_rate"]
        + df["shipping_mode"].map(modes) * 0.5
        - 0.6 * (df["priority"] == "Express")
        + 0.4 * df["order_month"].isin([11, 12]).astype(int)
        + rng.normal(0, 0.6, n)
    )
    prob = 1 / (1 + np.exp(-score))
    df[TARGET] = (rng.random(n) < prob).astype(int)

    df.to_csv(DATA_FILE, index=False)
    print(f"Sample data created: {DATA_FILE} ({n} shipments, "
          f"{df[TARGET].mean():.1%} delayed)")


# ---------------------------------------------------------------------------
# 2. Model building
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )


def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline(
            [("prep", build_preprocessor()),
             ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]
        ),
        "Random Forest": Pipeline(
            [("prep", build_preprocessor()),
             ("clf", RandomForestClassifier(
                 n_estimators=300, min_samples_leaf=3,
                 class_weight="balanced", random_state=42, n_jobs=-1))]
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }


# ---------------------------------------------------------------------------
# 3. Reporting
# ---------------------------------------------------------------------------
def feature_importance(model, top_n: int = 12) -> pd.Series:
    names = model.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    imp = pd.Series(model.named_steps["clf"].feature_importances_, index=names)
    return imp.sort_values(ascending=False).head(top_n)


def save_report(model, X_test, y_test) -> None:
    cm = confusion_matrix(y_test, model.predict(X_test))
    imp = feature_importance(model)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xticks([0, 1], ["On time", "Delayed"])
    axes[0].set_yticks([0, 1], ["On time", "Delayed"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

    axes[1].barh(imp.index[::-1], imp.values[::-1], color="#3b6ea5")
    axes[1].set_title("Top Delay Drivers (feature importance)")
    axes[1].set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(REPORT_FILE, dpi=130)
    plt.close()


# ---------------------------------------------------------------------------
# 4. Prediction for a single shipment
# ---------------------------------------------------------------------------
def predict_shipment(shipment: dict, model_path: str = MODEL_FILE) -> tuple:
    """Return (delay_probability, risk_label) for one shipment."""
    model = joblib.load(model_path)
    prob = float(model.predict_proba(pd.DataFrame([shipment]))[0, 1])
    label = "HIGH" if prob >= 0.65 else "MEDIUM" if prob >= 0.40 else "LOW"
    return prob, label


EXAMPLE_SHIPMENT = {
    "carrier": "BudgetShip",
    "shipping_mode": "Sea",
    "origin_region": "East",
    "destination_region": "West",
    "distance_km": 6500,
    "weight_kg": 450.0,
    "priority": "Standard",
    "order_month": 12,
    "day_of_week": 4,
    "weather_severity": 8.0,
    "traffic_index": 7.5,
    "is_holiday_week": 1,
    "carrier_past_delay_rate": 0.35,
}


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Shipment Delay Predictor")
    parser.add_argument("--generate-sample", action="store_true",
                        help="create a demo shipments.csv")
    parser.add_argument("--predict", action="store_true",
                        help="score an example shipment with the saved model and exit")
    args = parser.parse_args()

    if args.predict:
        if not os.path.exists(MODEL_FILE):
            raise SystemExit("No trained model found. Run the script without --predict first.")
        prob, label = predict_shipment(EXAMPLE_SHIPMENT)
        print("Example shipment:", EXAMPLE_SHIPMENT)
        print(f"\nDelay probability: {prob:.1%}  ->  risk: {label}")
        return

    if args.generate_sample or not os.path.exists(DATA_FILE):
        generate_sample_data()

    df = pd.read_csv(DATA_FILE)
    X, y = df[CATEGORICAL + NUMERIC], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train and compare models
    results, trained = {}, {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        trained[name] = model
        results[name] = evaluate(model, X_test, y_test)

    print("\n=== Model Comparison (test set) ===")
    print(pd.DataFrame(results).T.round(3).to_string())

    # The Random Forest is saved as the final model: it handles non-linear
    # effects and gives feature importances for the delay-driver chart.
    rf = trained["Random Forest"]
    print("\nClassification report (Random Forest):")
    print(classification_report(y_test, rf.predict(X_test),
                                target_names=["On time", "Delayed"]))
    print("Top delay drivers:")
    print(feature_importance(rf, top_n=8).round(3).to_string())

    save_report(rf, X_test, y_test)
    joblib.dump(rf, MODEL_FILE)
    print(f"\nSaved: {MODEL_FILE}, {REPORT_FILE}")

    prob, label = predict_shipment(EXAMPLE_SHIPMENT)
    print(f"\nExample high-risk shipment -> delay probability {prob:.1%} ({label} risk)")


if __name__ == "__main__":
    main()
