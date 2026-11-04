# Shipment Delay Predictor

Predicts whether a shipment will arrive late and shows which factors drive delays.
Trains Logistic Regression and Random Forest models, compares them, and saves the Random Forest.

## Setup
```
pip install -r requirements.txt
```

## Run
```
python shipment_delay_predictor.py --generate-sample   # demo data + train
python shipment_delay_predictor.py                     # train on your own shipments.csv
python shipment_delay_predictor.py --predict           # score an example shipment
```

## Input format (shipments.csv)
| Column | Description |
|---|---|
| carrier, shipping_mode, origin_region, destination_region, priority | Categorical |
| distance_km, weight_kg | Shipment size |
| order_month, day_of_week, is_holiday_week | Timing |
| weather_severity, traffic_index | Conditions (0-10) |
| carrier_past_delay_rate | Carrier's historical late rate (0-1) |
| is_delayed | Target: 1 = late, 0 = on time |

## Outputs
- `delay_model.joblib`: trained model
- `model_report.png`: confusion matrix and top delay drivers
- Console: accuracy, precision, recall, F1, ROC-AUC

## Predict from your own code
```python
from shipment_delay_predictor import predict_shipment
prob, risk = predict_shipment({...})   # same fields as the input columns, without is_delayed
```
