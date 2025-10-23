import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# --- Configuración de Página ---
st.set_page_config(page_title="Reconocimiento de Imágenes", layout="centered")

# --- Estilos CSS ---
st.markdown("""
<style>
    /* Centrar el título */
    h1 {
        text-align: center;
        color: #2c3e50;
    }
    
    /* Estilo para la imagen estática */
    .static-image img {
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    /* Estilo para el widget de cámara */
    [data-testid="stCameraInput"] {
        border-radius: 12px;
        border: 2px dashed #bdc3c7;
        padding: 10px;
    }
    
    /* Contenedor de la app */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Subtítulos */
    .stSubheader {
        text-align: center;
        color: #34495e;
        font-style: italic;
    }

    /* Versión de Python (mover al fondo) */
    [data-testid="stText"] {
        text-align: center;
        font-size: 0.8rem;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

# --- Cargar Modelo y Datos ---
# Advertencia de caché para el modelo
@st.cache_resource
def load_keras_model():
    try:
        return load_model('keras_model.h5')
    except Exception as e:
        st.error(f"Error al cargar 'keras_model.h5': {e}")
        st.error("Asegúrate de que el archivo del modelo esté en el mismo directorio.")
        return None

model = load_keras_model()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# --- Título y Descripción ---
st.title("Reconocimiento de Imágenes  ˶ᵔ ᵕ ᵔ˶ ")
st.subheader("Usando un modelo entrenado en Teachable Machine puedes usar esta app para identificar señas! Puede identificar un thumbs up, un thumbs down, y un peace sign.")

# --- Imagen Estática ---
try:
    image = Image.open('gray poodle dog peace sign emoji.jpeg')
    # Aplicar clase CSS a la imagen
    st.markdown('<div class="static-image">', unsafe_allow_html=True)
    st.image(image, width=350)
    st.markdown('</div>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("No se pudo encontrar la imagen 'OIG5.jpg'. Colócala en el directorio o elimina esta línea.")

st.write("---") # Separador

# --- Entrada de Cámara ---
img_file_buffer = st.camera_input("Toma una Foto para Analizar")

if model is None:
    st.stop() # Detener la ejecución si el modelo no se cargó

if img_file_buffer is not None:
    # Procesar la imagen
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)
    
    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    # Normalizar la imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    # Cargar la imagen en el array
    data[0] = normalized_image_array

    # --- Inferencia y Resultados ---
    try:
        prediction = model.predict(data)
        
        st.write("---") # Separador
        st.subheader("Resultados de la Predicción:")

        prob_thumbs_up = prediction[0][0]
        prob_thumbs_down = prediction[0][1]
        prob_peace = prediction[0][2]
        
        # Banderas para saber si se mostró algo
        detected = False

        if prob_thumbs_up > 0.5:
          st.success(f'👍 Thumbs up, con Probabilidad: {prob_thumbs_up*100:.2f}%')
          detected = True
        
        if prob_thumbs_down > 0.5:
          st.error(f'👎 Thumbs down, con Probabilidad: {prob_thumbs_down*100:.2f}%')
          detected = True
        
        if prob_peace > 0.5:
          st.info(f'✌️ Peace, con Probabilidad: {prob_peace*100:.2f}%')
          detected = True
        
        # Si ninguna probabilidad superó el 0.5
        if not detected:
            st.warning("No se reconoció ninguna seña con alta probabilidad. ¡Intenta de nuevo!")

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")

# Mover la versión de Python al final
st.text(f"Versión de Python: {platform.python_version()}")

