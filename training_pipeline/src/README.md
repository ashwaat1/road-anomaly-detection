# Source Code

## Data Pipeline (`data_pipeline/`)

Scripts for downloading, auditing, converting, and merging nine road damage datasets into a unified 75,000+ image YOLO-format corpus.

| Script | Purpose |
|---|---|
| `download_datasets.py` | Download raw datasets from Kaggle via kagglehub |
| `audit_datasets.py` | Read-only structural audit of downloaded datasets |
| `convert_xml_to_yolo.py` | Convert PascalVOC XML annotations (DS4, DS5) to YOLO format |
| `merge_datasets.py` | Merge 5 datasets into unified `merged_road_damage/` directory |
| `download_additional.py` | Download and merge 2 additional datasets (DS8, DS9) |
| `add_negative_samples.py` | Add undamaged road images as negative samples |

**Execution order:** download -> audit -> convert -> merge -> download_additional -> add_negative_samples

## Training (`training/`)

Multi-stage progressive training pipeline.

| Script | Stage | Epochs | Key Focus |
|---|---|---|---|
| `train_stage1_base.py` | Base training | 60 | Core feature learning with MuSGD, mosaic augmentation |
| `train_stage2_augment.py` | Augmentation fine-tune | 40 | DDP training, YOLO-native augmentations |
| `train_stage3_augment_alb.py` | Albumentations training | 40 | Custom Albumentations pipeline |
| `hard_negative_mining.py` | Data expansion | - | Augment hard samples from validation |
| `train_stage4_robust.py` | Robustness fine-tune | 20 | Shadow/blur/noise pipeline, frozen backbone |

## Evaluation (`evaluation/`)

| Script | Purpose |
|---|---|
| `evaluate_weather.py` | Weather robustness evaluation across 5 simulated conditions |
