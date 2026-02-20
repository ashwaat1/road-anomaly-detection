# Results

## Test Evaluation (`test_evaluation/`)

Outputs from the custom evaluation script on the 9,471-image test set.

| File | Description |
|---|---|
| `summary.json` | Headline metrics (mAP, precision, recall, F1, speed) |
| `per_class_results.csv` | Per-class AP, precision, recall, F1 |
| `per_image_results.csv` | Per-image TP/FP/FN/F1 (9,471 rows) |
| `conf_sweep.csv` | Precision/recall at 50 confidence thresholds |
| `confusion_matrix.png` | 5-class confusion matrix |
| `pr_curve.png` | Precision-recall curves per class |
| `f1_confidence_curve.png` | F1 vs. confidence threshold |
| `per_class_metrics.png` | Bar chart of per-class AP |
| `size_analysis.png` | Detection accuracy by object size (small/medium/large) |
| `inference_speed.png` | Latency distribution histogram |
| `confidence_histogram.png` | Prediction confidence distribution |
| `iou_distribution.png` | IoU distribution of true positives |

## Weather Evaluation (`weather_evaluation/`)

Outputs from weather robustness evaluation across 5 simulated conditions.

| File | Description |
|---|---|
| `weather_summary.json` | Per-condition and per-class metrics |
| `weather_comparison.csv` | Tabular condition comparison |
| `weather_comparison.png` | Side-by-side bar chart |
| `weather_delta.png` | Performance delta from normal baseline |

## Training Runs (`training_runs/`)

Ultralytics training outputs including loss curves, sample predictions, and model checkpoints.
