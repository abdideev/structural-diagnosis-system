import pandas as pd
import os
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
# Ruta absoluta base
BASE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_SAVE_PATH = os.path.join(BASE_OUTPUT_DIR, "nlp_finetuned")
# Ruta relativa al dataset (ajusta si es necesario)
DATASET_PATH = os.path.join(PROJECT_ROOT, 'data', 'datasets', 'fisuras_dataset.csv') 

def train_nlp_model():
    print("="*60)
    print(" INICIANDO ENTRENAMIENTO: FINE-TUNING (NLP)")
    print("="*60)

    # 1. Cargar CSV
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: No se encuentra {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)
    
    # Asegurar formato
    if 'label' not in df.columns or 'text' not in df.columns:
        print("❌ Error: El CSV debe tener columnas 'label' y 'text'")
        return

    # Mapear etiquetas
    label_map = {"corrosion": 0, "punzonamiento": 1}
    df["label"] = df["label"].map(label_map)
    # Filtrar nulos por si acaso
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    # 2. Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 3. Tokenización
    model_name = "distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Preparar columnas para PyTorch
    tokenized_train = tokenized_train.remove_columns(["text", "__index_level_0__"])
    tokenized_test = tokenized_test.remove_columns(["text", "__index_level_0__"])
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")
    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")

    # 4. Configurar Modelo
    id2label = {0: "corrosion", 1: "punzonamiento"}
    label2id = {"corrosion": 0, "punzonamiento": 1}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=id2label, label2id=label2id
    )

    # 5. Argumentos de Entrenamiento
    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_OUTPUT_DIR, "checkpoints_nlp"),
        
        eval_strategy="epoch",  
        
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    # 6. Entrenar
    trainer.train()

    # 7. Evaluar
    eval_results = trainer.evaluate()
    print(f"📊 Resultados finales NLP: {eval_results}")

    # 8. Guardar
    print(f"💾 Guardando modelo en: {MODEL_SAVE_PATH}")
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("✅ Fine-tuning completado.")

if __name__ == "__main__":
    train_nlp_model()