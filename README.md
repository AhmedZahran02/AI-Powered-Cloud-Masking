# AI-Powered Cloud Masking for Satellite Imagery

![Project Banner](media/image1.png)  
_Cloud masking visualization example_

## 📌 Project Overview

This repository contains an end-to-end solution for **automated cloud detection in satellite imagery** using machine learning. The system processes multi-spectral satellite data to generate accurate cloud masks, addressing a critical challenge in remote sensing applications.

## 🚀 Key Features

- **Multi-model Architecture**: Implements both deep learning (UNet, DeepLabV3+) and classical (Random Forest) approaches
- **Advanced Preprocessing**: Handles 4-band satellite imagery with specialized normalization
- **Competition-Ready**: Fully compliant with Kaggle-style submission requirements
- **Reproducible**: Docker support and detailed experiment tracking

## 🛠 Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended for training)
- [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive) and [cuDNN 8.1](https://developer.nvidia.com/cudnn) (for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/[your-username]/cloud-masking.git
cd cloud-masking

# Install dependencies
pip install -r requirements.txt

# Download dataset (place in data/raw/)
gdown "https://drive.google.com/uc?id=1-cU2qx7XY_lwCC7PKOnnNRkeyRto80gC"
unzip dataset.zip -d data/raw/
```

## 🏗 Project Structure

```
cloud-masking/
├── data/
│   ├── raw/                  # Original dataset (to be downloaded)
│   ├── processed/            # Processed data after augmentation/normalization
│   ├── train/                # Training split
│   │   ├── images/           # Training images
│   │   └── masks/            # Corresponding masks
│   └── val/                  # Validation split
│       ├── images/           # Validation images
│       └── masks/            # Corresponding masks
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_classical_model_experiments.ipynb
│   └── 04_deep_learning_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration parameters
│   ├── data_loader.py        # Data loading and augmentation
│   ├── preprocessing.py      # Image preprocessing utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py           # UNet implementation
│   │   ├── deeplab.py        # DeepLabV3 implementation
│   │   ├── random_forest.py  # Classical model
│   │   └── model_utils.py    # Model utilities
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation metrics
│   ├── utils.py              # Helper functions
│   └── visualization.py      # Visualization utilities
├── outputs/
│   ├── models/               # Saved model weights
│   ├── logs/                 # Training logs
│   └── predictions/          # Prediction outputs
├── run_inference.py          # Inference script for test set
├── train.py                  # Main training script
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
├── README.md                 # Project documentation
└── report/                   # Final report assets
    ├── figures/              # Report figures
    └── Report.pdf # Final PDF report
```

## 🧠 Model Performance

| Model         | Dice Coefficient | Inference Time (512px) | Parameters |
| ------------- | ---------------- | ---------------------- | ---------- |
| UNet          | 0.92             | 15ms                   | 7.8M       |
| DeepLabV3+    | 0.91             | 18ms                   | 41.3M      |
| Random Forest | 0.85             | 8ms                    | -          |

_Metrics on validation set (RTX 3080 GPU)_

## 💻 Usage

### Training

```bash
# Train UNet model (default)
python train.py

# Train specific model
python train.py --model deeplab --epochs 50 --batch_size 32
```

### Inference

```bash
# Generate competition submission
python run_inference.py \
    --test_dir path/to/test_images \
    --output_csv submission.csv \
    --model_path outputs/models/unet_best.h5
```

### Docker Support

```bash
# Build and run container
docker build -t cloud-masking .
docker run --gpus all -v $(pwd)/data:/app/data cloud-masking
```

## 📊 Sample Results

![Prediction Examples](media/image3.png)  
_Left: Input image | Middle: Ground truth | Right: Model prediction_

## 📚 Documentation

- [Full Project Report](report/ST-Project-Report.pdf)
- [API Reference](docs/API.md)
- [Competition Guidelines](docs/COMPETITION.md)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Cairo University Faculty of Engineering for project supervision
- ESA Copernicus Program for sample datasets
- TensorFlow and PyTorch communities for open-source tools
