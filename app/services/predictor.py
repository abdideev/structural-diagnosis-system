import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class FissurePredictor:
    def __init__(self):
        # RUTA BASE ABSOLUTA (Ajustada a tu carpeta)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
        self.models_dir = os.path.join(self.root_dir, 'models')
        
        print(f"--- CARGANDO MODELOS DESDE: {self.models_dir} ---")
        
        # 1. Cargar CNN
        cnn_path = os.path.join(self.models_dir, 'cnn_model.h5')
        if os.path.exists(cnn_path):
            self.cnn_model = tf.keras.models.load_model(cnn_path)
            print(f"✅ CNN cargada: {cnn_path}")
        else:
            raise FileNotFoundError(f"❌ No se encontró CNN en: {cnn_path}")

        # 2. Cargar ID3 y sus Columnas (Híbrido)
        id3_path = os.path.join(self.models_dir, 'id3_classifier.pkl')
        cols_path = os.path.join(self.models_dir, 'id3_columns.pkl')
        
        if os.path.exists(id3_path) and os.path.exists(cols_path):
            self.id3_model = joblib.load(id3_path)
            self.id3_columns = joblib.load(cols_path)
            print(f"✅ Modelo ID3 cargado: {id3_path}")
        else:
            raise FileNotFoundError(f"❌ Faltan archivos del ID3 en: {self.models_dir}")

        # 3. Cargar Fine-Tuning (NLP) - Opcional para que no rompa si falta
        nlp_path = os.path.join(self.models_dir, 'nlp_finetuned')
        try:
            self.nlp_tokenizer = AutoTokenizer.from_pretrained(nlp_path)
            self.nlp_model = AutoModelForSequenceClassification.from_pretrained(nlp_path)
            print(f"✅ Modelo NLP cargado: {nlp_path}")
        except Exception as e:
            print(f"⚠️ No se cargó modelo NLP (funcionará sin texto): {e}")
            self.nlp_model = None

    def preprocess_image(self, image_path):
        """Prepara la imagen para la CNN (150x150, RGB, /255.0)"""
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0

    def predict(self, image_path, structural_data, text_description):
        """
        Ejecuta el pipeline separado:
        1. Rama Visual/Estructural -> CNN + ID3
        2. Rama Textual -> DistilBERT
        """
        results = {}

        # ==========================================
        # RAMA 1: SISTEMA HÍBRIDO (CNN + ID3)
        # ==========================================
        
        # A. Obtener probabilidades visuales de la CNN
        img_proc = self.preprocess_image(image_path)
        cnn_preds = self.cnn_model.predict(img_proc, verbose=0)
        
        # Asumimos: Clase 0 = Corrosion, Clase 1 = Punzonamiento
        # (Verifica esto con tu train_cnn.py, class_names)
        prob_corrosion = float(cnn_preds[0][0])
        prob_punz = float(cnn_preds[0][1]) if len(cnn_preds[0]) > 1 else 1.0 - prob_corrosion

        # B. Preparar datos para el ID3
        # Unimos la info estructural del usuario con las probs de la CNN
        input_row = structural_data.copy()
        input_row['cnn_prob_corrosion'] = prob_corrosion
        input_row['cnn_prob_punzonamiento'] = prob_punz

        # Convertimos a DataFrame
        df_input = pd.DataFrame([input_row])

        # One-Hot Encoding y Alineación de Columnas
        # Esto asegura que si el usuario puso "viga", coincida con la columna "ubicacion_viga"
        df_encoded = pd.get_dummies(df_input)
        
        # Crear un DF final con EXACTAMENTE las columnas que aprendió el ID3
        df_final = pd.DataFrame(columns=self.id3_columns)
        
        # Llenar los valores coincidentes, el resto en 0
        for col in df_final.columns:
            if col in df_encoded.columns:
                df_final.at[0, col] = df_encoded.at[0, col]
            else:
                df_final.at[0, col] = 0
        
        # Predicción ID3
        id3_pred = self.id3_model.predict(df_final.fillna(0))[0]
        id3_label = "Punzonamiento" if id3_pred == 1 else "Corrosión"

        results['hibrido'] = {
            'diagnostico': id3_label,
            'cnn_probs': {'Corrosion': prob_corrosion, 'Punzonamiento': prob_punz}
        }

        # ==========================================
        # RAMA 2: SISTEMA DE TEXTO (NLP)
        # ==========================================
        nlp_res = "Modelo no disponible"
        nlp_score = 0.0
        
        if self.nlp_model and text_description:
            inputs = self.nlp_tokenizer(
                text_description, return_tensors="pt", truncation=True, max_length=128
            )
            with torch.no_grad():
                outputs = self.nlp_model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                # Asumimos ID 0 = Corrosion, ID 1 = Punzonamiento (ver train_finetuning.py)
                score_punz = float(probs[0][1])
                score_corr = float(probs[0][0])
                
            if score_punz > score_corr:
                nlp_res = "Punzonamiento"
                nlp_score = score_punz
            else:
                nlp_res = "Corrosión"
                nlp_score = score_corr
        
        results['texto'] = {
            'diagnostico': nlp_res,
            'score': nlp_score,
            'input_text': text_description
        }

        return results