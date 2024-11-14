import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import ssd300_vgg16
import numpy as np
import time
import plotly.express as px
import pandas as pd

# Configurar el ícono y el título de la página
st.set_page_config(page_title="Mosquito Detector", page_icon="mosquito.ico")

# Estilos personalizados
st.markdown("""
<style>
    .title {
        color: #f1c40f;
        font-size: 32px;
        font-weight: bold;
    }
    .model-title {
        color: #2375a6;
        font-size: 30px;
        font-weight: bold;
        margin-top: 20px;
    }
    .result-box {
        background-color: #eaf2f8;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
    }
    .stApp {
        background-color: #183458;
    }
</style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.markdown('<h1 class="title">Clasificación de Imágenes de Mosquitos 🦟</h1>', unsafe_allow_html=True)

# Cargar modelo YOLOv5
@st.cache_resource
def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/YOLOv5/best.pt', force_reload=True)
    model.eval()
    return model.to('cpu')

yolo_model = load_yolo_model()

# Cargar modelo SSD
@st.cache_resource
def load_ssd_model():
    model = ssd300_vgg16(pretrained=False)
    model.load_state_dict(torch.load("models/SSD/ssd_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

ssd_model = load_ssd_model()

# Función para medir el tiempo de inferencia
def measure_inference_time(model, image_tensor):
    start_time = time.time()
    with torch.no_grad():
        model(image_tensor)
    end_time = time.time()
    return end_time - start_time

# Cargar imagen para predicción
uploaded_file = st.file_uploader("Elige una imagen de mosquito para evaluación...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Cargada')

    # Preprocesamiento para ambos modelos
    transform = transforms.Compose([transforms.Resize((300, 300)), transforms.ToTensor()])
    img_tensor = transform(image).unsqueeze(0)

    # Medir tiempos de inferencia
    yolo_time = measure_inference_time(yolo_model, image)
    ssd_time = measure_inference_time(ssd_model, img_tensor)

    # Mostrar resultados de tiempo de inferencia
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write(f"⏱️ **Tiempo de Inferencia YOLOv5:** {yolo_time:.4f} segundos")
    st.write(f"⏱️ **Tiempo de Inferencia SSD300 VGG16:** {ssd_time:.4f} segundos")
    st.markdown('</div>', unsafe_allow_html=True)

    # Datos para la gráfica
    performance_data = {
        "Modelo": ["YOLOv5", "SSD300 VGG16"],
        "Tiempo de Inferencia (s)": [yolo_time, ssd_time]
    }
    df_performance = pd.DataFrame(performance_data)

    # Checkbox para mostrar/ocultar la gráfica
    show_chart = st.checkbox("Mostrar gráfica de rendimiento")

    if show_chart:
        # Crear gráfico interactivo de barras usando Plotly
        fig = px.bar(
            df_performance,
            x="Modelo",
            y="Tiempo de Inferencia (s)",
            color="Modelo",
            title="Comparativa de Tiempo de Inferencia entre YOLOv5 y SSD300 VGG16",
            text="Tiempo de Inferencia (s)"
        )
        fig.update_layout(xaxis_title="Modelo", yaxis_title="Tiempo de Inferencia (s)")
        fig.update_traces(texttemplate='%{text:.4f}s', textposition='outside')

        # Mostrar gráfico interactivo
        st.plotly_chart(fig)

# Evaluación con el modelo YOLOv5
uploaded_file_yolo = st.file_uploader("Elige una imagen para YOLOv5...", type=["jpg", "jpeg", "png"], key="yolo")
if uploaded_file_yolo is not None:
    image = Image.open(uploaded_file_yolo)
    results = yolo_model(image)
    img_with_boxes_yolo = np.squeeze(results.render())

    col1, col2 = st.columns(2)
    col1.image(image, caption='Imagen Original')
    col2.image(img_with_boxes_yolo, caption='Resultado YOLOv5')

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown("🔍 **Predicciones YOLOv5:**", unsafe_allow_html=True)
    for i, (box, conf, cls) in enumerate(zip(results.xyxy[0], results.xyxyn[0][:, 4], results.xyxyn[0][:, 5])):
        st.write(f"📍 Objeto {i+1}: Clase {int(cls)} con {conf:.2f} de confianza")
    st.markdown('</div>', unsafe_allow_html=True)

# Evaluación con el modelo SSD
uploaded_file_ssd = st.file_uploader("Elige una imagen para SSD...", type=["jpg", "jpeg", "png"], key="ssd")
if uploaded_file_ssd is not None:
    image = Image.open(uploaded_file_ssd)
    st.image(image, caption='Imagen Cargada')

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        predictions = ssd_model(img_tensor)

    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write("🔍 **Predicciones del modelo SSD:**", unsafe_allow_html=True)
    for i, (box, label, score) in enumerate(zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores'])):
        if score > 0.5:
            st.write(f"📍 Objeto {i+1}: Clase {label} con {score:.2f} de confianza")
    st.markdown('</div>', unsafe_allow_html=True)
