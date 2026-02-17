# Sistema Experto de Diagnóstico de Fisuras Estructurales 🤖

---

## Objetivo 🎯

El objetivo de este proyecto fue desarrollar un **Sistema Híbrido de Inteligencia Artificial** capaz de clasificar y diagnosticar patologías en estructuras de concreto (vigas y columnas). A diferencia de los sistemas tradicionales, este software utiliza una **arquitectura de decisión conjunta** que combina análisis visual, datos físicos y descripciones textuales para proporcionar evaluaciones estructurales integrales.

---

## Características ✨

Este sistema experto incluye las siguientes capacidades:

- **Análisis Visual**: Red Neuronal Convolucional (CNN) para reconocimiento de patrones de fisuras
- **Sistema de Decisión Experto**: Árbol de Decisión ID3 que combina probabilidades visuales con mediciones físicas
- **Procesamiento de Lenguaje Natural**: Modelo DistilBERT afinado que analiza descripciones textuales
- **Predicción Híbrida**: Orquestación de tres modelos proporcionando diagnósticos multi-perspectiva
- **Interfaz Gráfica Interactiva**: Aplicación de escritorio construida con Flet para interacción fluida
- **Salida Integral**: Diagnóstico técnico con puntuaciones de confianza y análisis contextual

---

## Arquitectura del Sistema 🧠

El sistema procesa la información a través de dos flujos paralelos orquestados por el núcleo de predicción (`predictor.py`):

### 1. Flujo Híbrido (Visual + Estructural)
* **Paso A (CNN):** Una Red Neuronal Convolucional analiza la imagen de la fisura y extrae probabilidades (ej. *85% Corrosión*)
* **Paso B (ID3):** Un Árbol de Decisión (Algoritmo ID3) toma estas probabilidades junto con datos físicos (ancho de fisura, ubicación) para emitir el diagnóstico técnico final

### 2. Flujo Independiente (NLP)
* **Paso C (LLM):** Un modelo de lenguaje (DistilBERT Afinado) analiza la descripción textual proporcionada por el usuario para ofrecer una "segunda opinión" basada en el contexto narrativo

---

## Stack Tecnológico 💻

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flet](https://img.shields.io/badge/Flet-UI-purple)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-orange)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-ID3-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-NLP-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

**Tecnologías Utilizadas:**
- **Lenguaje:** Python 3.10+
- **Framework GUI:** Flet (Flutter para Python)
- **Machine Learning:** TensorFlow (Keras), Scikit-Learn
- **NLP:** HuggingFace Transformers, PyTorch
- **Procesamiento de Datos:** Pandas, NumPy

---

## Estructura del Proyecto 📂
```text
├── app/
│   ├── main.py              # 🚀 PUNTO DE ENTRADA: Ejecuta la interfaz gráfica
│   ├── services/
│   │   └── predictor.py     # 🧠 MOTOR LÓGICO: Carga los 3 modelos y orquesta la predicción
│   └── views/               # Componentes de la interfaz (Carga, Resultados)
├── data/                    # Conjuntos de datos (Imágenes y CSVs)
├── models/                  # Carpeta para modelos entrenados (.h5, .pkl)
├── training/                # Scripts de entrenamiento (Ejecutar una vez)
│   ├── train_cnn.py         # Entrena el modelo de Visión Artificial
│   ├── train_expert.py      # Entrena el Árbol de Decisión
│   └── train_finetuning.py  # Entrena el Modelo de Lenguaje
└── requirements.txt         # Dependencias del proyecto
```

---

## Instalación 🚀

Sigue estos pasos para desplegar el proyecto en tu entorno local.

### 1. Clonar y Configurar
```bash
git clone https://github.com/paulomantilla04/structural-diagnosis-system.git
cd structural-diagnosis-system

# Crear entorno virtual (Recomendado)
python -m venv .venv

# Activar en Windows:
.venv\Scripts\activate

# Activar en Mac/Linux:
source .venv/bin/activate
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Generación de Modelos (CRÍTICO) ⚠️

Este repositorio no incluye los archivos pesados de los modelos. Debes generarlos en tu máquina ejecutando los scripts de entrenamiento en el siguiente orden estricto:

#### A. Entrenar la CNN (Visión):
```bash
python training/train_cnn.py
```
**Genera:** `models/cnn_model.h5`

#### B. Entrenar el Sistema Experto (ID3):
```bash
python training/train_expert.py
```
**Genera:** `models/id3_classifier.pkl`

#### C. Entrenar el NLP (Texto):
*(Requiere conexión a internet para descargar DistilBERT)*
```bash
python training/train_finetuning.py
```
**Genera:** `models/nlp_finetuned/`

---

## Uso 💡

Una vez generados los modelos, inicia el sistema ejecutando:
```bash
python app/main.py
```

1. **Carga:** El sistema cargará los 3 modelos en memoria (puede tardar unos segundos)
2. **Interfaz:** Se abrirá una ventana de escritorio
3. **Diagnóstico:** Sube una imagen, completa los campos de datos físicos y escribe una descripción. Presiona "Analizar" para ver el resultado conjunto

---

## Requisitos 📝

- Python 3.10 o superior
- GPU compatible con CUDA (recomendado para inferencia CNN más rápida)
- Mínimo 8GB de RAM
- Conexión a internet (para descargas iniciales de modelos)

---

**Desarrollado con ❤️ por Abdiel 🕷️**
