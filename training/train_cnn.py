# models/training/train_cnn.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE RUTAS ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Usamos 'r' antes de las comillas para que Python interprete las barras invertidas correctamente
DATASET_PATH = os.path.join(PROJECT_ROOT, 'data', 'images')

# Ruta base donde se guardarán los resultados
BASE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'models')

# Nombre específico del archivo del modelo
MODEL_FILENAME = "cnn_model.h5"
MODEL_SAVE_PATH = os.path.join(BASE_OUTPUT_PATH, MODEL_FILENAME)

# Configuración de entrenamiento
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
VAL_SPLIT = 0.2
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE

def get_augmentation():
    return models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2)
    ])

def cargar_dataset(path):
    # Verificar que la ruta existe antes de intentar cargar
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ La ruta de las imágenes no existe: {path}")

    train_ds = image_dataset_from_directory(
        path,
        validation_split=VAL_SPLIT,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    val_ds = image_dataset_from_directory(
        path,
        validation_split=VAL_SPLIT,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    class_names = train_ds.class_names
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names

def crear_modelo(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        get_augmentation(),
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_history(history, base_path):
    acc = history.history['accuracy']
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history['loss']
    val_loss = history.history.get('val_loss', [])
    epochs = range(len(acc))
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.legend()
    plt.title("Precisión")
    
    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.legend()
    plt.title("Pérdida")
    plt.tight_layout()
    
    # Guardar en la misma carpeta models
    out_path = os.path.join(base_path, "training_history_cnn.png")
    plt.savefig(out_path, dpi=150)
    print(f"📊 Gráfica guardada en: {out_path}")

def main():
    # Asegurar que la carpeta models exista
    os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)
    
    print(f"📂 Cargando imágenes desde: {DATASET_PATH}")
    try:
        train_ds, val_ds, classes = cargar_dataset(DATASET_PATH)
        print(f"✅ Clases encontradas: {classes}")
        
        model = crear_modelo(num_classes=len(classes))
        model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]

        print("🚀 Iniciando entrenamiento...")
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
        
        print(f"💾 Guardando modelo en: {MODEL_SAVE_PATH}")
        model.save(MODEL_SAVE_PATH)
        
        plot_history(history, BASE_OUTPUT_PATH)
        print("✨ Entrenamiento CNN finalizado con éxito.")
        
    except Exception as e:
        print(f"❌ Error crítico: {e}")

if __name__ == "__main__":
    main()