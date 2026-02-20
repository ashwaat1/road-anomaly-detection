# AI SOC - Road Anomaly Detection

Real-time road damage detection using YOLO26n, designed for edge deployment on a Raspberry Pi. The model detects five classes of road surface anomalies from dashcam video at >= 5 FPS.

## Classes

| ID | Class | Description |
|:---:|---|---|
| 0 | `pothole` | Circular / irregular surface cavities |
| 1 | `longitudinal_crack` | Cracks parallel to travel direction |
| 2 | `transverse_crack` | Cracks perpendicular to travel direction |
| 3 | `alligator_crack` | Interconnected / patterned cracking |
| 4 | `other_damage` | Manholes, speed bumps, unsurfaced road, repairs |

## Key Results

| Metric | Value |
|---|---|
| mAP@0.50 | 0.7104 |
| mAP@0.50:0.95 | 0.4310 |
| Best F1 | 0.6661 (conf=0.270) |
| Inference speed (GPU) | 56.4 FPS |

## Repository Structure

```
.
|-- best.pt                          # Trained YOLO26n weights
|-- src/
|   |-- data_pipeline/               # Dataset download, audit, conversion, merge
|   |-- training/                    # Multi-stage training scripts
|   |-- evaluation/                  # Test set and weather robustness evaluation
|-- results/
|   |-- test_evaluation/             # Metrics, plots, CSVs from test set eval
|   |-- weather_evaluation/          # Weather robustness charts and data
|   |-- training_runs/               # Checkpoints and training curves
```

## Reproducing the Pipeline

### 1. Dataset Preparation

Run scripts in this order from `src/data_pipeline/`:

```
python download_datasets.py          # Fetch raw datasets from Kaggle
python audit_datasets.py             # Structural audit (read-only)
python convert_xml_to_yolo.py        # PascalVOC XML to YOLO format
python merge_datasets.py             # Merge into unified directory
python download_additional.py        # Fetch and merge DS8, DS9
python add_negative_samples.py       # Add undamaged-road images
```

### 2. Training

Run scripts in order from `src/training/`:

```
python train_stage1_base.py              # Stage 1: Base training (60 epochs, MuSGD)
python train_stage2_augment.py           # Stage 2: Augmentation fine-tune (40 epochs, AdamW + DDP)
python train_stage3_augment_alb.py       # Stage 3: Albumentations augmentation (40 epochs)
python hard_negative_mining.py           # Hard negative mining pass
python train_stage4_robust.py            # Stage 4: Robustness fine-tune (20 epochs, frozen backbone)
```

### 3. Evaluation

```
python src/evaluation/evaluate_weather.py    # Weather robustness evaluation
```

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- Ultralytics (YOLO26 support)
- Albumentations
- kagglehub
