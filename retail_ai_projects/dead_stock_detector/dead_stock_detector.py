"""
Dead Stock Detector
-------------------
Identifies dead and slow-moving inventory from sales history and
suggests an action for each product.

Usage:
    python dead_stock_detector.py --generate-sample      # create demo CSVs
    python dead_stock_detector.py                        # analyse inventory.csv + sales.csv
    python dead_stock_detector.py --dead-days 120 --slow-days 60

Input files:
    inventory.csv : sku, product_name, category, stock_qty, unit_cost
    sales.csv     : date, sku, qty_sold

Outputs:
    dead_stock_report.csv   : full per-SKU classification
    dead_stock_summary.png  : charts (status counts + locked-up value)
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt

INVENTORY_FILE = "inventory.csv"
SALES_FILE = "sales.csv"
REPORT_FILE = "dead_stock_report.csv"
CHART_FILE = "dead_stock_summary.png"


# ---------------------------------------------------------------------------
# 1. Sample data generator
# ---------------------------------------------------------------------------
def generate_sample_data(n_products: int = 60, seed: int = 42) -> None:
    """Create a demo inventory.csv and sales.csv."""
    rng = np.random.default_rng(seed)
    categories = ["Electronics", "Clothing", "Grocery", "Home", "Toys", "Sports"]

    skus = [f"SKU{1000 + i}" for i in range(n_products)]
    inventory = pd.DataFrame(
        {
            "sku": skus,
            "product_name": [f"Product {i + 1}" for i in range(n_products)],
            "category": rng.choice(categories, n_products),
            "stock_qty": rng.integers(5, 400, n_products),
            "unit_cost": np.round(rng.uniform(2, 150, n_products), 2),
        }
    )

    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=365)
    days = pd.date_range(start, end, freq="D")

    # Each product gets a behaviour: healthy, slow, or dead (stopped selling)
    behaviours = rng.choice(["healthy", "slow", "dead"], n_products, p=[0.55, 0.25, 0.20])
    rows = []
    for sku, behaviour in zip(skus, behaviours):
        if behaviour == "healthy":
            rate, last_active = rng.uniform(1.5, 6), end
        elif behaviour == "slow":
            rate, last_active = rng.uniform(0.05, 0.3), end
        else:  # dead: sold well once, then stopped 100-300 days ago
            rate = rng.uniform(0.5, 3)
            last_active = end - pd.Timedelta(days=int(rng.integers(100, 300)))

        for d in days[days <= last_active]:
            qty = rng.poisson(rate)
            if qty > 0:
                rows.append((d.date(), sku, int(qty)))

    pd.DataFrame(rows, columns=["date", "sku", "qty_sold"]).to_csv(SALES_FILE, index=False)
    inventory.to_csv(INVENTORY_FILE, index=False)
    print(f"Sample data created: {INVENTORY_FILE}, {SALES_FILE}")


# ---------------------------------------------------------------------------
# 2. Detection logic
# ---------------------------------------------------------------------------
def classify_inventory(
    inventory: pd.DataFrame,
    sales: pd.DataFrame,
    dead_days: int = 90,
    slow_days: int = 45,
    cover_limit: int = 365,
    window: int = 90,
) -> pd.DataFrame:
    """Return inventory with metrics, a status and a recommended action."""
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    ref_date = sales["date"].max()  # "today" for the analysis

    # Last sale date per SKU
    last_sale = sales.groupby("sku")["date"].max().rename("last_sale_date")

    # Units sold in the recent window
    recent = sales[sales["date"] > ref_date - pd.Timedelta(days=window)]
    units_recent = recent.groupby("sku")["qty_sold"].sum().rename("units_sold_recent")

    df = inventory.merge(last_sale, on="sku", how="left").merge(
        units_recent, on="sku", how="left"
    )
    df["units_sold_recent"] = df["units_sold_recent"].fillna(0)

    # Metrics
    df["days_since_last_sale"] = (ref_date - df["last_sale_date"]).dt.days
    df["days_since_last_sale"] = df["days_since_last_sale"].fillna(9999)  # never sold
    df["avg_daily_sales"] = df["units_sold_recent"] / window
    df["days_of_cover"] = np.where(
        df["avg_daily_sales"] > 0, df["stock_qty"] / df["avg_daily_sales"], np.inf
    )
    df["stock_value"] = (df["stock_qty"] * df["unit_cost"]).round(2)

    # Classification rules
    def status(row):
        if row["stock_qty"] == 0:
            return "Out of Stock"
        if row["days_since_last_sale"] >= dead_days:
            return "Dead"
        if row["days_since_last_sale"] >= slow_days or row["days_of_cover"] > cover_limit:
            return "Slow-Moving"
        return "Active"

    df["status"] = df.apply(status, axis=1)

    actions = {
        "Dead": "Liquidate: deep discount, bundle, or return to supplier",
        "Slow-Moving": "Run promotion and stop reordering",
        "Active": "No action needed",
        "Out of Stock": "Check demand and reorder if needed",
    }
    df["recommended_action"] = df["status"].map(actions)

    df["days_of_cover"] = df["days_of_cover"].replace(np.inf, np.nan).round(1)
    df["avg_daily_sales"] = df["avg_daily_sales"].round(3)
    return df.sort_values(["status", "stock_value"], ascending=[True, False])


# ---------------------------------------------------------------------------
# 3. Reporting
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby("status")
        .agg(products=("sku", "count"), stock_value=("stock_value", "sum"))
        .sort_values("stock_value", ascending=False)
    )
    total_value = df["stock_value"].sum()
    dead_value = df.loc[df["status"] == "Dead", "stock_value"].sum()

    print("\n=== Inventory Health Summary ===")
    print(summary.to_string(float_format=lambda x: f"{x:,.2f}"))
    print(f"\nTotal inventory value : {total_value:,.2f}")
    print(f"Capital stuck in dead stock : {dead_value:,.2f} "
          f"({dead_value / total_value:.1%})")

    print("\nTop 10 dead-stock items by value:")
    top = df[df["status"] == "Dead"].nlargest(10, "stock_value")
    print(
        top[["sku", "product_name", "category", "stock_qty",
             "days_since_last_sale", "stock_value"]].to_string(index=False)
    )


def save_chart(df: pd.DataFrame) -> None:
    order = ["Active", "Slow-Moving", "Dead", "Out of Stock"]
    colors = {"Active": "#2e8b57", "Slow-Moving": "#f0ad4e",
              "Dead": "#d9534f", "Out of Stock": "#999999"}

    counts = df["status"].value_counts().reindex(order).fillna(0)
    values = df.groupby("status")["stock_value"].sum().reindex(order).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(counts.index, counts.values, color=[colors[s] for s in counts.index])
    axes[0].set_title("Products by Status")
    axes[0].set_ylabel("Number of products")

    axes[1].bar(values.index, values.values, color=[colors[s] for s in values.index])
    axes[1].set_title("Stock Value by Status")
    axes[1].set_ylabel("Value")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=130)
    plt.close()


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Dead Stock Detector")
    parser.add_argument("--generate-sample", action="store_true",
                        help="create demo inventory.csv and sales.csv")
    parser.add_argument("--dead-days", type=int, default=90,
                        help="days without a sale to call an item dead (default 90)")
    parser.add_argument("--slow-days", type=int, default=45,
                        help="days without a sale to call an item slow-moving (default 45)")
    args = parser.parse_args()

    if args.generate_sample or not (os.path.exists(INVENTORY_FILE) and os.path.exists(SALES_FILE)):
        generate_sample_data()

    inventory = pd.read_csv(INVENTORY_FILE)
    sales = pd.read_csv(SALES_FILE)

    result = classify_inventory(inventory, sales,
                                dead_days=args.dead_days, slow_days=args.slow_days)
    result.to_csv(REPORT_FILE, index=False)
    save_chart(result)
    print_summary(result)
    print(f"\nSaved: {REPORT_FILE}, {CHART_FILE}")


if __name__ == "__main__":
    main()
