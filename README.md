# Emotion Classification with Custom Transformers

This project implements a transformer-based emotion classifier trained on the [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) dataset, developed by Google Research. The dataset contains 200k+ Reddit comments labeled with 28 emotions. The goal is to train, evaluate, and compare different transformer configurations from scratch — without relying on pretrained models.

## 🚀 Highlights

- Built a custom transformer encoder with configurable depth, width, and attention heads.
- Trained and compared 12 different architectures using accuracy and AUC as metrics.
- Visualized ROC curves and confusion matrices to assess emotion-level performance.
- Achieved ~0.84 macro-AUC and ~41% accuracy on a 28-class classification task.


## 🧠 Key Components

- **Data**: GoEmotions (`https://huggingface.co/datasets/google-research-datasets/go_emotions` via 🤗 Datasets)
- **Model**: PyTorch-based transformer encoder
- **Evaluation**: Accuracy, macro-AUC, ROC curves, and confusion matrices

## 📊 Model Results

| Model (L-H-D)     | Val Accuracy | Val AUC |
|------------------|--------------|---------|
| 1-2-256          | 41.1%        | 0.8416  |
| 2-2-256          | 40.6%        | 0.8449  |
| 2-4-256          | 40.9%        | 0.8446  |

The best-performing model (`L=2, H=4, D=256`) was further evaluated on the test set:

- **🧪 Test Accuracy**: `0.4078`
- **🧪 Test AUC (macro)**: `0.8409`

These results confirm the model's strong ranking ability and its consistency across validation and test sets in a challenging 28-class emotion classification task.

## 📈 Visualizations

- ROC curve comparison across top models
- Confusion matrices per model to visualize common misclassifications

## 🧪 To Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/akandath/Emotion-Text-Classification
   cd emotion-transformer

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Notebook:
    ```bash
    jupyter notebook classifier.ipynb

## 🙏 Acknowledgments

- **GoEmotions Dataset** by [Google Research](https://huggingface.co/datasets/google-research-datasets/go_emotions) for providing the annotated Reddit comments used in this project.
- **Hugging Face 🤗** for their open-source `datasets` and `transformers` libraries, which made data handling and tokenization seamless.
- **Carnegie Mellon University** for the academic environment and coursework that inspired this deep dive into custom transformer architectures.
- **PyTorch** for providing the flexible deep learning framework used to build and train all models from scratch.

## 📬 Contact Me

If you have questions, suggestions, or would like to collaborate, feel free to reach out:

**Ashwin Kandath**  
📧 ash.kandath@gmail.com 
🔗 [LinkedIn](https://www.linkedin.com/in/ashwinkandath/)