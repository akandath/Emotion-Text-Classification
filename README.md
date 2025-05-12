# Emotion Classification with Custom Transformers

This project implements a transformer-based emotion classifier trained on the [GoEmotions](https://github.com/google-research/goemotions) dataset, developed by Google Research. The dataset contains 200k+ Reddit comments labeled with 28 emotions. The goal is to train, evaluate, and compare different transformer configurations from scratch — without relying on pretrained models.

## 🚀 Highlights

- Built a custom transformer encoder with configurable depth, width, and attention heads.
- Trained and compared 12 different architectures using accuracy and AUC as metrics.
- Visualized ROC curves and confusion matrices to assess emotion-level performance.
- Achieved ~0.84 macro-AUC and ~41% accuracy on a 28-class classification task.


## 🧠 Key Components

- **Data**: GoEmotions (`google-research-datasets/go_emotions` via 🤗 Datasets)
- **Model**: PyTorch-based transformer encoder
- **Evaluation**: Accuracy, macro-AUC, ROC curves, and confusion matrices

## 📊 Model Results

| Model (L-H-D)     | Val Accuracy | Val AUC |
|------------------|--------------|---------|
| 1-2-256          | 41.1%        | 0.8416  |
| 2-2-256          | 40.6%        | 0.8449  |
| 2-4-256          | 40.9%        | 0.8446  |

## 📈 Visualizations

- ROC curve comparison across top models
- Confusion matrices per model to visualize common misclassifications

## 🧪 To Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/akandath/emotion-transformer.git
   cd emotion-transformer

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Notebook:
    ```bash
    jupyter notebook classifier.ipynb

