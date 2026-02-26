# 🎭 Emotion Detection from Text using DistilBERT

> **Beyond Positive/Negative — Detecting 6 Human Emotions in Real-Time using NLP & Deep Learning**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat&logo=huggingface)
![Gradio](https://img.shields.io/badge/Gradio-Dashboard-orange?style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=flat&logo=pytorch)
![Colab](https://img.shields.io/badge/Google%20Colab-Ready-green?style=flat&logo=googlecolab)

---

## 🚀 Project Overview

Most sentiment analysis tools only tell you **Positive** or **Negative**.

This project goes **deeper** — it detects the actual human emotion behind any text, choosing from **6 emotion classes:**

| Emotion | Label | Example |
|---------|-------|---------|
| 😢 Sadness | `sadness` | *"I miss my best friend so much"* |
| 🤩 Joy | `joy` | *"I just got my dream job!"* |
| ❤️ Love | `love` | *"I adore spending time with my family"* |
| 😠 Anger | `anger` | *"How dare they do this to me!"* |
| 😨 Fear | `fear` | *"I'm terrified of what comes next"* |
| 😲 Surprise | `surprise` | *"I can't believe this just happened!"* |

---

## 🧠 How It Works

```
User Input Text
      ↓
DistilBERT Tokenizer  →  Converts text to tokens
      ↓
Fine-tuned DistilBERT  →  Learns emotion patterns
      ↓
Softmax Layer  →  Converts to probabilities
      ↓
Gradio Dashboard  →  Shows real-time results!
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `DistilBERT` | Pre-trained transformer model (fine-tuned) |
| `HuggingFace Transformers` | Model & tokenizer loading |
| `HuggingFace Datasets` | `dair-ai/emotion` dataset (20k samples) |
| `PyTorch` | Deep learning framework |
| `Gradio` | Real-time interactive dashboard |
| `Scikit-learn` | Evaluation metrics |
| `Google Colab` | Training environment (free GPU) |

---

## 📊 Dataset

- **Source:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) on HuggingFace
- **Size:** ~20,000 labeled sentences
- **Classes:** 6 emotions (sadness, joy, love, anger, fear, surprise)
- **Split:** 16,000 train / 2,000 validation / 2,000 test

---

## ⚡ Quick Start

### Run on Google Colab (Recommended)
1. Open the notebook in Google Colab
2. Enable GPU: `Runtime > Change runtime type > T4 GPU`
3. Run all cells from top to bottom
4. Get a live public dashboard link via Gradio!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](your_colab_link_here)

### Install Dependencies Locally
```bash
pip install datasets transformers torch scikit-learn gradio pandas
```

---

## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~92% |
| F1 Score (weighted) | ~91% |
| Training Epochs | 3 |
| Batch Size | 32 |

---

## 📁 Project Structure

```
emotion-detection-nlp/
│
├── Emotion_Detection.ipynb   # Main Colab notebook (train + dashboard)
├── README.md                 # You are here!
└── requirements.txt          # All dependencies
```

---

## 💡 What I Learned

- How to load and use datasets from **HuggingFace Hub**
- How **transformer models** (DistilBERT) work under the hood
- Fine-tuning a pre-trained model for custom classification
- Building real-time ML dashboards with **Gradio**
- Evaluating models using confusion matrices and F1 scores

---

## 🔮 Future Improvements

- [ ] Deploy on HuggingFace Spaces (permanent public URL)
- [ ] Add multilingual emotion detection
- [ ] Train on larger datasets for better accuracy
- [ ] Add voice/audio input support

---

## 👨‍💻 About Me

I'm a final year **AI & ML student** passionate about building real-world NLP applications. This project is part of my journey to go beyond textbook knowledge and build things that actually work!

📫 Connect with me on [LinkedIn](www.linkedin.com/in/vedantamrajyalakshmi)

---

⭐ **If you found this helpful, give it a star!** It motivates me to build more! 🚀
