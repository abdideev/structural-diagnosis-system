"""
Utilidades compartidas para preprocesamiento de imágenes y datos.
"""

import cv2
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

# ----------------------------------------
# PREPROCESAMIENTO DE IMÁGENES
# ----------------------------------------

def cargar_y_preprocesar_imagen(ruta_imagen, target_size=(150, 150), color_mode='rgb'):
    """
    Carga una imagen y la preprocesa para el modelo CNN.
    
    Args:
        ruta_imagen: Ruta al archivo de imagen
        target_size: Tupla (altura, ancho) para redimensionar
        color_mode: 'rgb' o 'grayscale'
    
    Returns:
        numpy.ndarray: Imagen preprocesada lista para el modelo
    """
    try:
        img = keras_image.load_img(
            ruta_imagen,
            color_mode=color_mode,
            target_size=target_size
        )
        img_array = keras_image.img_to_array(img)
        # No normalizamos aquí, el modelo tiene capa Rescaling integrada
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión batch
        return img_array
    except Exception as e:
        print(f"❌ Error al cargar imagen: {e}")
        return None

# ----------------------------------------
# ANÁLISIS MORFOLÓGICO DE FISURAS
# ----------------------------------------

def analizar_morfologia_fisura(ruta_imagen):
    """
    Analiza características morfológicas de una fisura.
    Útil para features adicionales o validación visual.
    
    Returns:
        dict: Diccionario con métricas (grosor promedio, máximo, mínimo)
    """
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None
    
    # Suavizado y binarización
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(img_blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invertir si es necesario
    if np.mean(img[binary == 1]) > np.mean(img[binary == 0]):
        binary = 1 - binary
    
    if np.sum(binary) == 0:
        return {
            'grosor_promedio': 0,
            'grosor_maximo': 0,
            'grosor_minimo': 0,
            'deteccion_exitosa': False
        }
    
    # Esqueleto y mapa de distancias
    skeleton = skeletonize(binary > 0)
    dist_map = distance_transform_edt(binary)
    
    if np.any(skeleton):
        thickness_values = dist_map[skeleton] * 2
        
        return {
            'grosor_promedio': float(np.mean(thickness_values)),
            'grosor_maximo': float(np.max(thickness_values)),
            'grosor_minimo': float(np.min(thickness_values)),
            'deteccion_exitosa': True
        }
    
    return {
        'grosor_promedio': 0,
        'grosor_maximo': 0,
        'grosor_minimo': 0,
        'deteccion_exitosa': False
    }

# ----------------------------------------
# VALIDACIÓN DE DATOS
# ----------------------------------------

def validar_probabilidades_cnn(prob_corrosion, prob_punzonamiento):
    """
    Valida que las probabilidades del CNN sean coherentes.
    
    Returns:
        bool: True si las probabilidades son válidas
    """
    suma = prob_corrosion + prob_punzonamiento
    return (
        0 <= prob_corrosion <= 1 and
        0 <= prob_punzonamiento <= 1 and
        abs(suma - 1.0) < 0.01  # Tolerancia de redondeo
    )

def normalizar_probabilidades(prob_corrosion, prob_punzonamiento):
    """Normaliza las probabilidades para que sumen exactamente 1.0"""
    suma = prob_corrosion + prob_punzonamiento
    if suma == 0:
        return 0.5, 0.5  # Caso extremo: reparto equitativo
    return prob_corrosion / suma, prob_punzonamiento / suma

# ----------------------------------------
# GENERACIÓN DE CSV DE EJEMPLO (OPCIONAL)
# ----------------------------------------

def generar_csv_ejemplo(ruta_salida='data/diagnosticos_ejemplo.csv', num_muestras=100):
    """
    Genera un CSV de ejemplo con datos simulados para pruebas.
    Útil si aún no tienes el dataset real.
    """
    import pandas as pd
    
    np.random.seed(42)
    
    data = {
        'cnn_prob_corrosion': [],
        'cnn_prob_punzonamiento': [],
        'ubicacion': [],
        'posicion_en_losa': [],
        'manchas_oxido': [],
        'concreto_desprendido': [],
        'ambiente_humedo': [],
        'DIAGNOSTICO_REAL': []
    }
    
    for _ in range(num_muestras):
        # Simular casos de corrosión (50% de las muestras)
        if np.random.rand() < 0.5:
            data['cnn_prob_corrosion'].append(np.random.uniform(0.6, 0.95))
            data['cnn_prob_punzonamiento'].append(1 - data['cnn_prob_corrosion'][-1])
            data['manchas_oxido'].append(np.random.choice(['si', 'no'], p=[0.8, 0.2]))
            data['concreto_desprendido'].append(np.random.choice(['si', 'no'], p=[0.7, 0.3]))
            data['ambiente_humedo'].append(np.random.choice(['si', 'no'], p=[0.7, 0.3]))
            data['DIAGNOSTICO_REAL'].append(0)
        else:
            # Simular casos de punzonamiento
            data['cnn_prob_punzonamiento'].append(np.random.uniform(0.6, 0.95))
            data['cnn_prob_corrosion'].append(1 - data['cnn_prob_punzonamiento'][-1])
            data['manchas_oxido'].append(np.random.choice(['si', 'no'], p=[0.2, 0.8]))
            data['concreto_desprendido'].append(np.random.choice(['si', 'no'], p=[0.5, 0.5]))
            data['ambiente_humedo'].append(np.random.choice(['si', 'no'], p=[0.4, 0.6]))
            data['DIAGNOSTICO_REAL'].append(1)
        
        data['ubicacion'].append(np.random.choice(['columna', 'viga', 'losa']))
        data['posicion_en_losa'].append(np.random.choice(['centro', 'borde', 'esquina']))
    
    df = pd.DataFrame(data)
    df.to_csv(ruta_salida, index=False)
    print(f"✅ CSV de ejemplo generado en '{ruta_salida}'")
    return df

# ----------------------------------------
# PRUEBAS UNITARIAS (OPCIONAL)
# ----------------------------------------

if __name__ == "__main__":
    print("Ejecutando pruebas de utilidades...")
    
    # Prueba de validación de probabilidades
    assert validar_probabilidades_cnn(0.7, 0.3) == True
    assert validar_probabilidades_cnn(1.2, -0.2) == False
    print("✅ Validación de probabilidades funciona correctamente")
    
    # Prueba de normalización
    norm = normalizar_probabilidades(0.6, 0.5)
    assert abs(sum(norm) - 1.0) < 0.001
    print("✅ Normalización de probabilidades funciona correctamente")
    
    print("\n✅ Todas las pruebas pasaron exitosamente")