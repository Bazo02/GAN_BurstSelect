
# GAN_BurstSelect

A Generative Adversarial Approach to Burst-Sequence Image Ranking.

This project uses a GAN-based model to automatically rank or generate the best image in a burst sequence, based on features like sharpness, blurriness, posture, and eye contact.

## Setup

### 1. Clone and set up environment

```bash
git clone https://github.com/Bazo02/GAN_BurstSelect.git
cd GAN_BurstSelect

# Create virtual environment (if not already created)
python3 -m venv .venv
source .venv/bin/activate  # On Mac/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
````

### 2. Configuration

Update paths and hyperparameters in `config.yaml`. Example values:

```yaml
img_height: 128
img_width: 128
latent_dim: 100
batch_size: 16
epochs: 50
models_dir: models
outputs_dir: outputs
synth_dir: Project_4630/Synthetic_Data/Images
labels: Project_4630/Synthetic_Data/pairwise_labels.json
```

##  Running the Project

### Train the GAN model

```bash
python src/train_gan.py --config config.yaml
```

### Evaluate the model

```bash
python src/evaluate.py --config config.yaml
```

### Generate example images from the trained generator

```bash
python src/generate_sample.py
```

Images will be saved in the `outputs/` folder.

## 📁 Folder Structure

```
GAN_BurstSelect/
├── src/                  # Source code
│   ├── train_gan.py
│   ├── evaluate.py
│   ├── generate_sample.py
│   └── ...
├── outputs/              # Generated samples
├── models/               # Saved models (.keras)
├── Project_4630/         # Raw and synthetic image data
├── config.yaml
├── requirements.txt
└── README.md
```

## Authors

* Alexander Bazo
* Kimberly Crimson
* Espen Fodstad
