# Dead Stock Detector

Finds dead and slow-moving inventory from sales history and recommends an action for each product.

## Setup
```
pip install -r requirements.txt
```

## Run
```
python dead_stock_detector.py --generate-sample   # demo data
python dead_stock_detector.py                     # use your own inventory.csv + sales.csv
python dead_stock_detector.py --dead-days 120 --slow-days 60
```

## Input format
- `inventory.csv`: sku, product_name, category, stock_qty, unit_cost
- `sales.csv`: date, sku, qty_sold

## Classification rules
| Status | Rule |
|---|---|
| Dead | No sale for >= dead-days (default 90), or never sold |
| Slow-Moving | No sale for >= slow-days (default 45), or days of cover > 365 |
| Active | Selling regularly |
| Out of Stock | stock_qty = 0 |

## Outputs
- `dead_stock_report.csv`: every SKU with metrics, status, recommended action
- `dead_stock_summary.png`: status counts and stock value charts
