# road-anomaly-detection
Inference, video processing and training pipelines for the "Road Anomaly Detection on Raspberry Pi" problem statement for the ARM Bharat AI-SOC challenge.

Members:
- Ashwaat Tarun T.S.
- Ram Bhatta
- Tanish Chudiwal

## 📁 Repository Structure

```text
.
├── training_pipeline/      # 3-stage progressive PyTorch training scripts
│   └── README.md           # Detailed training instructions and hyperparameters
├── quantization_pipeline/  # Scripts to convert the .pt model to INT8 ONNX
│   └── rdd_quantize.ipynb  # Kaggle notebook for calibration and export
├── inference_pipeline/     # Raspberry Pi 5 deployment scripts
│   ├── stream_sender.py    # Camera capture and RAM disk writer
│   └── yolo_detect.py      # RAM disk reader and YOLO inference engine
└── yolo26n_rdd_int8.onnx   # The final, deployment-ready quantized model

```

## Important Links

- Dataset: https://www.kaggle.com/datasets/tanishchudiwal0909/arm-ai-soc-dataset-merged
- Quantization: https://www.kaggle.com/code/rambhattaee22b047/quantisation-onnx, https://www.kaggle.com/code/rambhatta007/rdd-trainer

## 🧠 The Final Model

The root directory contains `yolo26n_rdd_int8.onnx`. This is the final production model. It has been trained on the RDD dataset using a 3-stage pipeline (Base, Augment, Robust) and statically quantized to INT8, specifically optimized for ARM Cortex-A76 SDOT hardware acceleration.

---

## 🚀 Pipeline Modules

### 1. Training Pipeline

The `training_pipeline/` directory contains the PyTorch scripts used to train the base model. It includes heavy geometric and lighting augmentations to ensure real-world robustness.
*Please refer to `training_pipeline/README.md` for complete reproduction steps.*

### 2. Quantization Pipeline

The `quantization_pipeline/` folder contains a standalone Jupyter/Kaggle Notebook.

* **How to run:** Upload this notebook to any free Kaggle session. It handles the data-reader calibration, applies Entropy-based static quantization, and specifically excludes the final detection head from 8-bit rounding to prevent "zero-detection" degradation.

### 3. Inference Pipeline (Raspberry Pi 5)

To bypass the bottleneck of running camera acquisition and AI inference in a single synchronous loop, the inference pipeline is split into two asynchronous scripts that communicate via a zero-copy RAM disk (POSIX shared memory).

* `stream_sender.py`: Initializes the Camera Module V3, pulls frames at hardware speed, and writes them directly to shared RAM.
* `yolo_detect.py`: Reads the latest frame from the RAM pointer, runs ONNX inference, applies a 5-frame temporal consensus filter to remove ghosting, and logs verified road damage.

---

## 🛠️ Running Inference on the Raspberry Pi 5

**Hardware Requirements:**

* Raspberry Pi 5 (Active Cooler / Heatsink recommended)
* Raspberry Pi Camera Module V3

**Prerequisites:**
Ensure you have your virtual environment set up with `onnxruntime`, `opencv-python`, `ultralytics`, and `picamera2` installed.

### Option A: Standard Terminal Execution

You will need to open two separate terminal tabs/windows connected to your Pi to run the asynchronous pipeline.

**Terminal Tab 1 (Start the Camera):**

```bash
cd inference_pipeline
python stream_sender.py

```

*Wait for the terminal to print that the stream is live and writing to RAM.*

**Terminal Tab 2 (Start the AI):**

```bash
# In a new tab/window
cd inference_pipeline
python yolo_detect.py

```

### Option B: Using `tmux` (Recommended for Headless/SSH)

If you are SSH'd into your Pi and want to run both scripts side-by-side in a single window without managing multiple SSH connections, use `tmux`.

1. Start a new tmux session:
```bash
tmux new -s dashcam

```


2. Split the window vertically by pressing `Ctrl+B`, then `%`.
3. In the **left pane**, start the camera:
```bash
cd inference_pipeline && python stream_sender.py

```


4. Switch to the **right pane** by pressing `Ctrl+B`, then `Right Arrow`.
5. Start the inference engine:
```bash
cd inference_pipeline && python yolo_detect.py

```



*(To detach and leave it running in the background, press `Ctrl+B`, then `D`.)*
