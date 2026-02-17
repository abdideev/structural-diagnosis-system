import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

# Ruta absoluta para guardar modelos
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models')
# Ruta relativa al dataset procesado (ajusta si tu csv está en otro lado)
CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'datasets', 'diagnosticos_improved.csv')

def train_expert_model():
    print("="*60)
    print(" INICIANDO ENTRENAMIENTO: MODELO EXPERTO (ID3 HÍBRIDO)")
    print("="*60)

    # 1. Cargar Datos
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: No se encuentra {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"✅ Dataset cargado: {df.shape[0]} registros.")

    # 2. Definir Features (X) y Target (y)
    # Usamos las columnas exactas de tu CSV 'diagnosticos_improved'
    target_col = 'DIAGNOSTICO_REAL'
    
    # Eliminamos el target para obtener X
    X_raw = df.drop(columns=[target_col])
    y = df[target_col]

    print("Features de entrada:", list(X_raw.columns))

    # 3. Preprocesamiento (One-Hot Encoding)
    # Convertimos texto (ej: 'viga') a números (viga=1, columna=0...)
    X_encoded = pd.get_dummies(X_raw)
    
    # Guardamos las columnas para usarlas en la predicción real
    final_features = list(X_encoded.columns)

    # 4. Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # 5. Entrenar ID3
    # criterion='entropy' es la clave para simular el algoritmo ID3
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    # 6. Evaluar
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 Precisión del modelo: {acc:.2%}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=['Corrosión', 'Punzonamiento']))

    # 7. Guardar Modelo y Columnas
    os.makedirs(BASE_MODEL_PATH, exist_ok=True)
    
    model_filename = os.path.join(BASE_MODEL_PATH, "id3_classifier.pkl")
    columns_filename = os.path.join(BASE_MODEL_PATH, "id3_columns.pkl")

    joblib.dump(clf, model_filename)
    joblib.dump(final_features, columns_filename)
    
    print(f"✅ Modelo guardado en: {model_filename}")
    print(f"✅ Metadatos guardados en: {columns_filename}")

    # (Opcional) Generar imagen del árbol
    try:
        plt.figure(figsize=(20,10))
        plot_tree(clf, feature_names=final_features, class_names=['Corrosión', 'Punzonamiento'], filled=True, fontsize=8)
        tree_img_path = os.path.join(BASE_MODEL_PATH, "id3_tree_viz.png")
        plt.savefig(tree_img_path)
        print(f"🖼️ Imagen del árbol guardada en: {tree_img_path}")
    except Exception as e:
        print(f"⚠️ No se pudo generar la imagen del árbol: {e}")

if __name__ == "__main__":
    train_expert_model()