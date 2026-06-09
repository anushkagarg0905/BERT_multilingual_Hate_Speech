# 🚀 Multilingual Hate Speech Detection System  
### *BERT • XLM-RoBERTa • Hybrid Deep Learning Architecture*

![Banner](https://img.shields.io/badge/AI-NLP-blue?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red?style=for-the-badge&logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge)

---

## 📌 Overview

This project presents a **robust multilingual hate speech detection system** leveraging transformer-based architectures and hybrid deep learning techniques.

The system classifies text into:
- **Hate Speech (1)**
- **Non-Hate Speech (0)**

---

## 🧠 Model Architecture

### 🔹 BERT (Baseline)
- `bert-base-uncased`
- Fine-tuned for classification
- Strong performance on English text

### 🔹 XLM-RoBERTa (Multilingual)
- `xlm-roberta-base`
- Handles cross-lingual inputs effectively

### 🔹 Hybrid Model (BERT + LSTM)
- Frozen BERT embeddings
- LSTM for sequential learning
- Dropout + Fully Connected layer

---

## 🏗️ System Pipeline

                                             Raw Text Data
                                                   ↓
                                           Text Normalization
                                                   ↓
                                       Dataset Merging & Cleaning
                                                   ↓
                                     Exploratory Data Analysis (EDA)
                                                   ↓
                                      Tokenization (BERT / XLM-R)
                                                   ↓
                                    Model Training (3 Architectures)
                                                   ↓
                                         Evaluation & Metrics
                                                   ↓
                                         Performance Comparison



---

## 📊 Dataset

### Sources:
- Code-Mixed Hinglish Hate Speech Detection Dataset from  Kaggle

### Preprocessing:
- Lowercasing  
- URL removal  
- Mention removal  
- Contraction expansion  
- Repeated character normalization  
- Special character cleaning  
- Label standardization  

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Framework:** PyTorch  
- **Models:** Hugging Face Transformers  
- **Libraries:**
  - pandas, numpy  
  - scikit-learn  
  - matplotlib, seaborn  
  - wordcloud  

---

## ⚡ Performance Optimization

- Mixed Precision Training (`torch.cuda.amp`)  
- GPU acceleration  
- Efficient batching  
- Learning rate scheduling  

---

## 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

**Metric Importance:**
- Recall → Detects harmful content  
- Precision → Avoids false accusations  
- F1 Score → Balanced performance  

---

## 📊 Results

| Model        | Accuracy | F1 Score | Notes |
|-------------|--------|---------|------|
| BERT        | ~0.85  | ~0.84   | Strong baseline |
| XLM-R       | ~0.88  | ~0.87   | Best multilingual performance |
| Hybrid      | ~0.83  | ~0.82   | Needs tuning |

> ⚠️ Values may vary depending on dataset sampling

---

## 📊 Visualizations

- Class Distribution  
- Text Length Distribution  
- WordCloud  
- Training vs Validation Loss  
- Accuracy Comparison  
- F1 Score Comparison  
- Confusion Matrices  

---

