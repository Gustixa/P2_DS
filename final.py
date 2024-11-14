import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import ssd300_vgg16
import numpy as np

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
</style>
""", unsafe_allow_html=True)

# Título de la app con estilo
st.markdown('<h1 class="title">Clasificación de Imágenes de Mosquitos 🦟</h1>', unsafe_allow_html=True)

# Introducción con estilo de subtítulo
st.markdown('<p class="subtitle">Cargue imágenes para identificar y clasificar especies de mosquitos de manera rápida y precisa 🔎</p>', unsafe_allow_html=True)

# Simulación de carrusel de imágenes
st.markdown('<h3 class="model-title">Conoce los diferentes tipos de mosquitos</h3>', unsafe_allow_html=True)

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
st.image(mosquito_images[index], caption=mosquito_names[index], use_container_width=True)

# Cargar modelo YOLOv5
@st.cache_resource
def load_yolo_model():
    model = torch.hub.load('yolov5', 'custom', path='models/YOLOv5/best.pt', source='local')
    model.eval()  # Configura el modelo en modo evaluación
    return model.to('cpu')  # Fuerza el uso de CPU

yolo_model = load_yolo_model()

# Configuración de la interfaz de Streamlit para SSD
st.markdown('<p class="model-title">Clasificación de Imágenes con modelo YOLOv5</p>', unsafe_allow_html=True)

# Configurar la sección de carga para YOLOv5
uploaded_file_yolo = st.file_uploader("Elige una imagen de mosquito para YOLOv5...", type=["jpg", "jpeg", "png"], key="yolo")
if uploaded_file_yolo is not None:
    image = Image.open(uploaded_file_yolo)
    
    # Realizar la predicción con YOLOv5 en CPU
    results = yolo_model(image)
    img_with_boxes_yolo = np.squeeze(results.render())  # Imagen con bounding boxes de YOLOv5
    
    # Crear columnas para mostrar la imagen original y con detección
    col1, col2 = st.columns(2)
    
    # Mostrar la imagen original en la primera columna
    col1.image(image, caption='Imagen Original', use_container_width=True)
    
    # Mostrar la imagen con detección de YOLOv5 en la segunda columna
    col2.image(img_with_boxes_yolo, caption='Resultado YOLOv5', use_container_width=True)
    
    # Mostrar detalles de las predicciones con fondo de resultado
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown("🔍 **Predicciones YOLOv5:**", unsafe_allow_html=True)
    for i, (box, conf, cls) in enumerate(zip(results.xyxy[0], results.xyxyn[0][:, 4], results.xyxyn[0][:, 5])):
        st.write(f"📍 Objeto {i+1}: Clase {int(cls)} con {conf:.2f} de confianza")
    st.markdown('</div>', unsafe_allow_html=True)

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
    st.image(image, caption='Imagen Cargada', use_container_width=True)

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
