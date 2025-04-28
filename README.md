# AI-Powered Cloud Masking for Satellite Imagery

## 📌 Project Overview

This repository contains an end-to-end solution for **automated cloud detection in satellite imagery** using machine learning and deep learning. The system processes multi-spectral satellite data to generate accurate cloud masks, addressing a critical challenge in remote sensing applications.

## 🚀 Key Features

- **Multi-model Architecture**: Implements both deep learning (UNet) and classical (Random Forest, SGD, Ensemble) approaches
- **Advanced Preprocessing**: Handles 4-band satellite imagery with specialized normalization
- **Competition-Ready**: Fully compliant with Kaggle-style submission requirements
- **Reproducible**: Docker support and detailed experiment tracking

## 🛠 Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended for training)
- [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive) (for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/AhmedZahran02/AI-Powered-Cloud-Masking.git
cd AI-Powered-Cloud-Masking

# Install dependencies
pip install -r requirements.txt

# Download dataset (place in data/raw/)
gdown "https://drive.google.com/uc?id=1-cU2qx7XY_lwCC7PKOnnNRkeyRto80gC"
unzip dataset.zip -d data/raw/
```

## 🏗 Project Structure

```
AI-POWERED-CLOUD-MASKING/
├── data/
│   ├── raw/                  # Original dataset
│   │   ├── data/             # images
│   │   └── masks/            # Corresponding masks
│   ├── processed/            # Processed data after augmentation/normalization
│   │   ├── data/             # Training images
│   │   │   ├── cloud_free/
│   │   │   ├── fully_clouded/
│   │   │   └── partially_clouded/
│   │   └── masks/            # Corresponding masks
│   │       ├── cloud_free/
│   │       ├── fully_clouded/
│   │       └── partially_clouded/
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_classical_model_experiments.ipynb     # Ensemble
│   ├── 03_classical_model_experiments_SGD.ipynb # SGD Model
│   └── 04_deep_learning_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration parameters
│   ├── data_loader.py        # Data loading and augmentation
│   ├── preprocessing.py      # Image preprocessing utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py           # UNet implementation
│   │   ├── slim_unet.py      # LightWeight UNet implementation
│   │   └── ensemble.py  # Classical model
│   ├── evaluate.py            # Evaluation metrics
│   ├── data_loader.py         # data loading
│   ├── config.py              # Configiration
│   ├── utils.py               # Helper functions
│   ├── rle-encoder-decoder.py # RLE encoder and decoder
│   └── visualization.py       # Visualization utilities
├── outputs/
│   ├── models/               # Saved model weights
│   ├── logs/                 # Training logs
│   │   └── model_logs.txt    # Records the size of the trained model and the number of operations
│   └── predictions/          # Prediction outputs
├── run_inference.py          # Inference script for test set
├── run_classical_inference.py          # Inference script for test set
├── test                      # Test Folder for new inference cases
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
├── evaluation_function.py    # evaluate on test set
├── submit_profile.py         # profile model pkl
├── README.md                 # Project documentation
├── Papers/                   # Research Papers
└── report/                   # Final report assets
    ├── figures/              # Report figures
    └── Report.pdf            # Final PDF report
```

## 🧠 Model Performance

| Model    | Dice Coefficient | Inference Time (512 x512) | Parameters |
| -------- | ---------------- | ------------------------- | ---------- |
| UNet     | 0.92             | -                         | 13.4M      |
| SlimUNet | 0.895            | -                         | 1.98M      |
| Ensemble | 0.85             | -                         | -          |

_Metrics on validation set (RTX 3060 GPU)_

## 💻 Usage

### Training

```bash
# Train UNet model (default)
Run the 04_deep_learning_experiments.ipynb notebook
```

### Inference

```bash
# Generate competition submission
python run_inference.py \
    --test_folder test
    --output_csv submission.csv \
    --model_path outputs/models/Slim_Unet_model.pth
    --threshold 0.39473684210526316
```

### Profiling

```bash
python submit_profile.py outputs/models/Looser_Slim_Unet_model.pkl 1 4 512 512
```

### Docker Support

```bash
# Build and run container
docker build -t cloud-masking .
docker run --gpus all -it cloud-masking
```

## 📊 Sample Results

_Left: Input image | Middle: Ground truth | Right: Model prediction_

## 📚 Documentation

- [Full Project Report](report/ST-Project-Report.pdf)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
