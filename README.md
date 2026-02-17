# Sistema Experto de Diagnóstico de Fisuras Estructurales 🏗️🤖

---

## Goal 🎯

The goal of this project was to develop a **Hybrid Artificial Intelligence System** capable of classifying and diagnosing pathologies in concrete structures (beams and columns). Unlike traditional systems, this software uses a **joint decision architecture** that combines visual analysis, physical data, and textual descriptions to provide comprehensive structural assessments.

---

## Features ✨

This expert system includes the following capabilities:

- **Visual Analysis**: Convolutional Neural Network (CNN) for crack pattern recognition
- **Expert Decision System**: ID3 Decision Tree combining visual probabilities with physical measurements
- **Natural Language Processing**: Fine-tuned DistilBERT model analyzing textual descriptions
- **Hybrid Prediction**: Three-model orchestration providing multi-perspective diagnostics
- **Interactive GUI**: Desktop application built with Flet for seamless user interaction
- **Comprehensive Output**: Technical diagnosis with confidence scores and contextual insights

---

## System Architecture 🧠

The system processes information through two parallel flows orchestrated by the prediction core (`predictor.py`):

### 1. Hybrid Flow (Visual + Structural)
* **Step A (CNN):** A Convolutional Neural Network analyzes the crack image and extracts probabilities (e.g., *85% Corrosion*)
* **Step B (ID3):** A Decision Tree (ID3 Algorithm) takes these probabilities along with physical data (crack width, location) to issue the final technical diagnosis

### 2. Independent Flow (NLP)
* **Step C (LLM):** A language model (Fine-Tuned DistilBERT) analyzes the textual description provided by the user to offer a "second opinion" based on narrative context

---

## Tech Stack 💻

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flet](https://img.shields.io/badge/Flet-UI-purple)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-orange)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-ID3-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-NLP-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

**Technologies Used:**
- **Language:** Python 3.10+
- **GUI Framework:** Flet (Flutter for Python)
- **Machine Learning:** TensorFlow (Keras), Scikit-Learn
- **NLP:** HuggingFace Transformers, PyTorch
- **Data Processing:** Pandas, NumPy

---

## Project Structure 📂
```text
├── app/
│   ├── main.py              # 🚀 ENTRY POINT: Runs the graphical interface
│   ├── services/
│   │   └── predictor.py     # 🧠 LOGIC ENGINE: Loads 3 models and orchestrates prediction
│   └── views/               # UI components (Upload, Results)
├── data/                    # Datasets (Images and CSVs)
├── models/                  # Folder for trained models (.h5, .pkl)
├── training/                # Training scripts (Run once)
│   ├── train_cnn.py         # Trains Computer Vision model
│   ├── train_expert.py      # Trains Decision Tree
│   └── train_finetuning.py  # Trains Language Model
└── requirements.txt         # Project dependencies
```

---

## Installation 🚀

Follow these steps to deploy the project in your local environment.

### 1. Clone and Configure
```bash
git clone https://github.com/paulomantilla04/structural-diagnosis-system.git
cd structural-diagnosis-system

# Create virtual environment (Recommended)
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on Mac/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Model Generation (CRITICAL) ⚠️

This repository does not include the heavy model files. You must generate them on your machine by running the training scripts in the following strict order:

#### A. Train the CNN (Vision):
```bash
python training/train_cnn.py
```
**Generates:** `models/cnn_model.h5`

#### B. Train the Expert System (ID3):
```bash
python training/train_expert.py
```
**Generates:** `models/id3_classifier.pkl`

#### C. Train the NLP (Text):
*(Requires internet connection to download DistilBERT)*
```bash
python training/train_finetuning.py
```
**Generates:** `models/nlp_finetuned/`

---

## Usage 💡

Once the models are generated, start the system by running:
```bash
python app/main.py
```

1. **Loading:** The system will load all 3 models into memory (may take a few seconds)
2. **Interface:** A desktop window will open
3. **Diagnosis:** Upload an image, complete the physical data fields, and write a description. Press "Analyze" to see the joint result

---

## Requirements 📝

- Python 3.10 or higher
- CUDA-compatible GPU (recommended for faster CNN inference)
- Minimum 8GB RAM
- Internet connection (for initial model downloads)

---

## Project URL 🔗

If you're interested in exploring this challenge or contributing, you can visit the repository:

**GitHub Repository:** [https://github.com/paulomantilla04/structural-diagnosis-system](https://github.com/paulomantilla04/structural-diagnosis-system)

---

## Future Enhancements 🔮

- Integration with real-time structural monitoring systems
- Mobile application version
- Extended pathology database
- Multi-language support
- Cloud-based deployment option

---

**Developed with ❤️ by Abdiel 🕷️**