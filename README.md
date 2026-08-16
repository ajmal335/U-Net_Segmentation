# U-Net Semantic Segmentation — Oxford-IIIT Pet Dataset

A professional PyTorch implementation of **U-Net for semantic image segmentation** using the **Oxford-IIIT Pet Dataset**.

The project covers the complete segmentation workflow: dataset preparation, custom PyTorch dataset loading, U-Net architecture, training, validation, evaluation, model checkpointing, and prediction visualization.

---

## 📌 Project Overview

Unlike image classification, where a model predicts a single class for an entire image, **semantic segmentation predicts a class for every pixel**.

In this project, U-Net learns to separate the **foreground pet from the background** at the pixel level.

### Example

**Input image**

🐕 / 🐈 Pet image

⬇️

**U-Net**

⬇️

**Segmentation mask**

* `1` → Pet / foreground
* `0` → Background

The predicted mask can then be used to isolate the pet from the rest of the image.

---

## 🎯 Objectives

The main goals of this project were to:

* Understand semantic segmentation
* Implement U-Net using PyTorch
* Build a custom segmentation dataset
* Work with image-mask pairs
* Implement encoder-decoder architecture
* Understand skip connections
* Train a segmentation model
* Calculate Dice Score
* Save the best model checkpoint
* Evaluate the model on unseen test data
* Generate segmentation predictions and visualizations

---

## 🧠 Model Architecture

The project uses the **U-Net architecture**, which consists of:

### Encoder

The encoder progressively reduces the spatial resolution of the image while learning increasingly meaningful features.

```text
Input Image
     ↓
Convolution
     ↓
Feature Extraction
     ↓
Downsampling
     ↓
Deeper Features
```

### Bottleneck

The bottleneck contains the deepest feature representation of the image.

```text
Encoder
   ↓
Bottleneck
   ↓
Decoder
```

### Decoder

The decoder gradually increases the spatial resolution to reconstruct a pixel-level segmentation mask.

### Skip Connections

U-Net uses skip connections between corresponding encoder and decoder levels.

These connections transfer spatial information from the encoder directly to the decoder, helping the model recover fine details and object boundaries.

```text
Encoder                    Decoder

  Feature ───────────────→ Feature
     ↓                       ↑
  Feature ───────────────→ Feature
     ↓                       ↑
  Feature ───────────────→ Feature
     ↓                       ↑
       ── Bottleneck ──
```

---

## 📂 Dataset

This project uses the **Oxford-IIIT Pet Dataset**.

The dataset contains:

* 7,390 images
* Pet images
* Pixel-level segmentation masks

### Dataset Split

| Split      |    Images |     Masks |
| ---------- | --------: | --------: |
| Train      |     5,173 |     5,173 |
| Validation |     1,108 |     1,108 |
| Test       |     1,109 |     1,109 |
| **Total**  | **7,390** | **7,390** |

Each image has a corresponding segmentation mask.

---

## 📁 Project Structure

```text
U-Net_Segmentation/
│
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   │
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   │
│   └── test/
│       ├── images/
│       └── masks/
│
├── dataset_download/
│   └── # Original downloaded dataset files
│
├── models/
│   └── best_model.pth
│
├── results/
│   ├── predictions/
│   └── metrics.txt
│
├── model.py
├── dataset.py
├── prepare_dataset.py
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Pillow
* Matplotlib
* tqdm

---

## 🐍 Environment

The project was developed using:

```text
Python 3.11
PyTorch
CPU
```

Training and evaluation were performed locally on CPU.

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/ajmal335/U-Net_Segmentation.git
```

```bash
cd U-Net_Segmentation
```

---

## 2. Create and activate the environment

Using Conda:

```bash
conda create -n ai-engineer python=3.11
```

Activate it:

```bash
conda activate ai-engineer
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Dataset Preparation

The original Oxford-IIIT Pet dataset was downloaded and processed into the following structure:

```text
dataset/
├── train/
│   ├── images/
│   └── masks/
│
├── val/
│   ├── images/
│   └── masks/
│
└── test/
    ├── images/
    └── masks/
```

The dataset preparation script performs the train/validation/test split and creates the corresponding image-mask pairs.

Run:

```bash
python prepare_dataset.py
```

Expected split:

```text
Train images: 5173
Val images:   1108
Test images:  1109
```

---

# 🧪 Dataset Verification

The custom dataset loader can be tested with:

```bash
python dataset.py
```

Example output:

```text
Dataset size: 5173
Image shape: torch.Size([3, 224, 224])
Mask shape : torch.Size([1, 224, 224])

Mask unique values:
tensor([0., 1.])
```

This verifies that:

* Images are loaded correctly
* Masks are loaded correctly
* Images have 3 channels
* Masks have 1 channel
* Masks contain binary foreground/background values

---

# 🏗️ Model Verification

The U-Net architecture can be tested using:

```bash
python model.py
```

Example:

```text
Input shape : torch.Size([2, 3, 224, 224])
Output shape: torch.Size([2, 1, 224, 224])
```

The model therefore produces a **pixel-level segmentation output** with the same spatial dimensions as the input.

---

# 🏋️ Training

The model is trained using:

* Binary segmentation
* Training loss
* Dice Score
* Validation evaluation
* Best-model checkpointing

Run:

```bash
python train.py
```

### Training Configuration

```text
Training images   : 5173
Validation images : 1108
Batch size        : 8
Epochs            : 5
Learning rate     : 0.001
Device            : CPU
```

### Training Result

The best validation Dice Score achieved during training was:

```text
Best validation Dice: 0.6807
```

The best model is saved to:

```text
models/best_model.pth
```

---

# 📊 Evaluation

The trained model was evaluated on the unseen test dataset.

Run:

```bash
python evaluate.py
```

### Test Results

```text
Test images : 1109

Test Dice Score : 0.6366
```

The evaluation results and visualization outputs are saved inside:

```text
results/
```

---

# 📈 Evaluation Metric

## Dice Score

The primary metric used in this project is the **Dice Score**, which measures the overlap between the predicted segmentation mask and the ground-truth mask.

A higher Dice Score indicates better overlap between prediction and ground truth.

The final test Dice Score obtained was:

```text
0.6366
```

This result demonstrates that the model learned meaningful foreground segmentation, although there is still room for improvement in accurately capturing object boundaries and fine details.

---

# 🔮 Prediction

The project also includes a prediction script for generating segmentation masks on new images.

Run:

```bash
python predict.py
```

The prediction workflow:

```text
Input Image
     ↓
Preprocessing
     ↓
Trained U-Net
     ↓
Pixel-level Prediction
     ↓
Segmentation Mask
     ↓
Visualization
```

Prediction visualizations are stored inside:

```text
results/
```

---

# 🖼️ Results

The `results/` directory contains the important outputs from model evaluation and prediction.

Example:

```text
results/
├── predictions/
│   ├── prediction_01.png
│   ├── prediction_02.png
│   └── ...
│
└── metrics.txt
```

The prediction visualizations are useful for inspecting how accurately the model identifies the foreground pet pixels.

---

# 🔍 What I Learned

Through this project, I practiced the complete semantic segmentation pipeline.

### Computer Vision

* Semantic segmentation
* Pixel-level classification
* Binary segmentation
* Image masks
* Foreground/background separation

### Deep Learning

* Encoder-decoder architectures
* Convolutional layers
* Downsampling
* Upsampling
* Skip connections
* Feature extraction

### PyTorch

* Custom `Dataset`
* `DataLoader`
* Model implementation
* Training loops
* Validation loops
* Model checkpointing
* Evaluation
* Prediction

### Model Evaluation

* Dice Score
* Qualitative prediction analysis
* Segmentation visualization

---

# ⚠️ Limitations

The current model achieves a test Dice Score of:

```text
0.6366
```

The predictions are not perfectly accurate around object boundaries and fine details.

Possible causes include:

* Limited training duration
* CPU-based training
* Basic U-Net architecture
* Image resizing
* Binary segmentation setup
* Limited hyperparameter tuning

The project is therefore considered a **baseline U-Net semantic segmentation implementation**, rather than a production-ready segmentation system.

---

# 🚀 Possible Improvements

Future improvements could include:

* Training for more epochs
* GPU training
* Data augmentation
* Better learning-rate scheduling
* Dice + Binary Cross Entropy loss
* More advanced U-Net variants
* Attention U-Net
* U-Net++
* DeepLabV3
* Improved image resolution
* Hyperparameter tuning
* More detailed boundary evaluation

---

# 📚 Project Workflow

The complete workflow is:

```text
Oxford-IIIT Pet Dataset
          ↓
Dataset Preparation
          ↓
Train / Validation / Test Split
          ↓
Custom PyTorch Dataset
          ↓
U-Net
          ↓
Training
          ↓
Validation
          ↓
Best Model Checkpoint
          ↓
Test Evaluation
          ↓
Prediction
          ↓
Segmentation Visualization
```

---

# 🎯 Project Status

```text
✅ Dataset downloaded
✅ Dataset prepared
✅ Train/validation/test split
✅ Custom Dataset
✅ U-Net architecture
✅ Model training
✅ Validation
✅ Best model checkpoint
✅ Test evaluation
✅ Prediction pipeline
✅ Results visualization
✅ GitHub-ready project
```

---

# 👨‍💻 Author

**Ajmal**

Aspiring AI Engineer

Focus areas:

* Machine Learning
* Deep Learning
* Computer Vision
* Natural Language Processing
* Large Language Models
* RAG
* AI Engineering

---

## ⭐ Next Step

This project is part of my AI Engineer learning roadmap.

The next stage of the roadmap is:

**Natural Language Processing (NLP)**

Topics include:

* Tokenization
* Word embeddings
* RNN
* LSTM
* GRU
* Transformers
* BERT
* Text classification

---

## 📄 License

This project is intended for educational and portfolio purposes.

The dataset is provided by the Oxford-IIIT Pet Dataset and is subject to its respective terms of use.
