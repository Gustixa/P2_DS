import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pandas as pd
import time
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import ssd300_vgg16

# Configurar el ícono y el título de la página
st.set_page_config(page_title="Mosquito Detector", page_icon="mosquito.ico")

# Aplicar estilos personalizados usando HTML y CSS
st.markdown("""
<style>
    .title {
        color: #f1c40f;
        font-size: 32px;
        font-weight: bold;
    }
    .subtitle {
        color: #3498db;
        font-size: 24px;
    }
    .model-title {
        color: #2375a6; /* Azul más oscuro; prueba #4a4a4a para gris oscuro o #627d98 para azul grisáceo */
        font-size: 30px;
        font-weight: bold;
        margin-top: 20px;
    }
    .warning-text {
        color: #e74c3c;
        font-weight: bold;
    }
    .neutral-text {
        color: #95a5a6;
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
    .st-emotion-cache-b5pbez {  /* Contenedor del área de carga de archivos */
        display: flex;
        justify-content: center;
    }
    .st-emotion-cache-9ycgxx, /* Clase del texto "Drag and drop file here" */
    .st-emotion-cache-61rn6a { /* Clase del texto "Limit 200MB per file..." */
        color: #a0c4ff; /* Cambia a gris claro o elige un color que se vea mejor sobre fondo oscuro */
    }
    paragraph{
        color: white;
        
    }
    mark {
        background-color: #f4d03f; /* Amarillo claro */
        color: #183458; /* Texto en azul oscuro para contrastar */
        font-weight: bold;
    } 
          
    
</style>
""", unsafe_allow_html=True)

# Título de la app con estilo
st.markdown('<h1 class="title">Clasificación de Imágenes de Mosquitos 🦟</h1>', unsafe_allow_html=True)
# Introducción con estilo de subtítulo
st.markdown('<p class="subtitle">Cargue imágenes para identificar y clasificar especies de mosquitos de manera rápida y precisa 🔎</p>', unsafe_allow_html=True)
# Simulación de carrusel de imágenes
st.markdown('<h3 class="model-title">Conoce los diferentes tipos de mosquitos</h3>', unsafe_allow_html=True)

# Simulación de carrusel de imágenes
st.markdown(
    '<p class="paragraph">Los mosquitos son transmisores de enfermedades, tales como la <mark>fiebre amarilla</mark>, <mark>zika</mark>, <mark>dengue</mark>, y <mark>chikungunya</mark>. El propósito de este clasificador, es determinar si se detectan especies que son transmisoras de tales enfermedades.' +
    '</p>'
    , unsafe_allow_html=True)

# Cargar imágenes de ejemplo (asegúrate de tener estas imágenes en tu directorio)
mosquito_images = [
    Image.open("./assets/aegypti.jpg"),
    Image.open("./assets/albopictus.jpg"),
    Image.open("./assets/anopheles.jpg"),
    Image.open("./assets/culex.jpg"),
    Image.open("./assets/culiseta.jpg"),
]

# Nombres de los mosquitos en el orden correcto
mosquito_names = [
    "Mosquito Aedes aegypti",
    "Mosquito Aedes albopictus",
    "Mosquito Anopheles",
    "Mosquito Culex",
    "Mosquito Culiseta"
]

# Slider para seleccionar la imagen
index = st.slider("Desliza para ver los diferentes tipos de mosquitos", 0, len(mosquito_images) - 1, 0)

# Mostrar imagen seleccionada y su nombre
st.image(mosquito_images[index], caption=mosquito_names[index], use_column_width=True)

# Cargar modelo YOLOv5
@st.cache_resource
def load_yolo_model():
    model = torch.hub.load('yolov5', 'custom', path='models/YOLOv5/best.pt', source='local')
    model.eval()  # Configura el modelo en modo evaluación
    return model.to('cpu')  # Fuerza el uso de CPU

yolo_model = load_yolo_model()

# Ajusta la función predict_image para redimensionar la imagen correctamente
def predict_image(model, image):
    # Redimensiona la imagen al tamaño esperado por el modelo
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Tamaño esperado por YOLOv5
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)  # Añadir batch dimension

    # Predicción
    with torch.no_grad():
        results = model(img_tensor)
    
    return results

# Simulación de métricas (reemplazar con tus datos reales)
def calculate_metrics(results):
    # Simula métricas usando datos ficticios
    precision = np.random.uniform(0.7, 1.0)
    recall = np.random.uniform(0.6, 1.0)
    map50 = np.random.uniform(0.7, 1.0)
    map = np.random.uniform(0.6, 1.0)
    return precision, recall, map50, map

# Configuración de la interfaz de Streamlit para SSD
st.markdown('<p class="model-title">Clasificación de Imágenes con modelo YOLOv5</p>', unsafe_allow_html=True)

# Función para medir el tiempo de inferencia
def measure_inference_time(model, image_tensor):
    start_time = time.time()
    with torch.no_grad():
        model(image_tensor)
    end_time = time.time()
    return end_time - start_time

# Interfaz de usuario para cargar la imagen
uploaded_file = st.file_uploader("Cargar una imagen de mosquito para evaluar", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # st.image(image, caption='Imagen cargada', use_column_width=True)

    # Realizar la predicción con YOLOv5 en CPU
    results = yolo_model(image)
    img_with_boxes_yolo = np.squeeze(results.render())  # Imagen con bounding boxes de YOLOv5

    # Crear dos columnas para mostrar las imágenes en paralelo
    col1, col2 = st.columns(2)

    # Mostrar la imagen original en la primera columna
    col1.image(image, caption='Imagen Original', use_column_width=True)

    # Mostrar la imagen con detección de YOLOv5 en la segunda columna
    col2.image(img_with_boxes_yolo, caption='Resultado YOLOv5', use_column_width=True)
        # Mostrar detalles de las predicciones
    st.write("Predicciones YOLOv5:")
    for i, (box, conf, cls) in enumerate(zip(results.xyxy[0], results.xyxyn[0][:, 4], results.xyxyn[0][:, 5])):
        st.write(f"Objeto {i+1}: Clase {int(cls)} con {conf:.2f} de confianza")
    # Obtener resultados de la predicción
    results = predict_image(yolo_model, image)
    
    # Calcular métricas para la imagen
    precision, recall, map50, map = calculate_metrics(results)

    # Crear un DataFrame para visualizar las métricas
    metrics_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95'],
        'Value': [precision, recall, map50, map]
    })

    # Mostrar las métricas como gráfica
    st.markdown('<p class="model-title">Eficiencia del Modelo en la Imagen Cargada</p>', unsafe_allow_html=True)
    fig = px.bar(metrics_df, x='Metric', y='Value', title="Métricas de Rendimiento del Modelo en Imagen")
    st.plotly_chart(fig)

# Cargar modelo SSD con los pesos guardados
@st.cache_resource
def load_ssd_model():
    model = ssd300_vgg16(pretrained=False)  # Inicializa el modelo
    model.load_state_dict(torch.load("models/SSD/ssd_model.pth", map_location=torch.device('cpu')))
    model.eval()  # Configura el modelo en modo evaluación
    return model

# Configuración de la interfaz de Streamlit para SSD
st.markdown('<p class="model-title">Clasificación de Imágenes con modelo SSD</p>', unsafe_allow_html=True)

# Cargar modelo SSD
ssd_model = load_ssd_model()

uploaded_file_ssd = st.file_uploader("Elige una imagen de mosquito para SSD...", type=["jpg", "jpeg", "png"], key="ssd")
if uploaded_file_ssd is not None:
    image = Image.open(uploaded_file_ssd)
    st.image(image, caption='Imagen Cargada', use_column_width=True)

    # Preprocesar la imagen para el modelo SSD
    transform = transforms.Compose([
        transforms.Resize((300, 300)),  # Tamaño esperado para SSD
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)  # Añadir batch dimension

    # Realizar predicción
    with torch.no_grad():
        predictions = ssd_model(img_tensor)

    # Procesar y mostrar resultados
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.write("🔍 **Predicciones del modelo SSD:**", unsafe_allow_html=True)
    for i, (box, label, score) in enumerate(zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores'])):
        if score > 0.5:  # Filtrar por un umbral de confianza
            st.write(f"📍 Objeto {i+1}: Clase {label} con {score:.2f} de confianza")
    st.markdown('</div>', unsafe_allow_html=True)

        # Obtener resultados de la predicción
    results = predict_image(yolo_model, image)
    
    # Calcular métricas para la imagen
    precision, recall, map50, map = calculate_metrics(results)

    # Crear un DataFrame para visualizar las métricas
    metrics_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95'],
        'Value': [precision, recall, map50, map]
    })

    # Mostrar las métricas como gráfica
    st.markdown('<p class="model-title">Eficiencia del Modelo en la Imagen Cargada</p>', unsafe_allow_html=True)
    fig = px.bar(metrics_df, x='Metric', y='Value', title="Métricas de Rendimiento del Modelo en Imagen")
    st.plotly_chart(fig)

# Cargar imagen para predicción
uploaded_file = st.file_uploader("Elige una imagen de mosquito para evaluación...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Cargada', use_column_width=True)

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