import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import ssd300_vgg16
import numpy as np


# Cargar modelo YOLOv5
@st.cache_resource
def load_yolo_model():
    model = torch.hub.load('yolov5', 'custom', path='models/YOLOv5/best.pt', source='local')
    model.eval()  # Configura el modelo en modo evaluación
    return model.to('cpu')  # Fuerza el uso de CPU

yolo_model = load_yolo_model()

uploaded_file_yolo = st.file_uploader("Elige una imagen de mosquito para YOLOv5...", type=["jpg", "jpeg", "png"], key="yolo")
if uploaded_file_yolo is not None:
    image = Image.open(uploaded_file_yolo)

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


# Cargar el modelo SSD con los pesos guardados
@st.cache_resource
def load_ssd_model():
    model = ssd300_vgg16(pretrained=False)  # Inicializa el modelo
    model.load_state_dict(torch.load("ssd_model.pth", map_location=torch.device('cpu')))
    model.eval()  # Configura el modelo en modo evaluación
    return model


# Configuración de la interfaz de Streamlit para SSD
st.title("Clasificación de Imágenes de Mosquitos con SSD")

# Cargar modelo SSD
ssd_model = load_ssd_model()

uploaded_file_ssd = st.file_uploader("Elige una imagen de mosquito para SSD...", type=["jpg", "jpeg", "png"], key="ssd")
if uploaded_file_ssd is not None:
    image = Image.open(uploaded_file_ssd)
    st.image(image, caption='Imagen Cargada', use_column_width=True)

    # Preprocesar la imagen
    transform = transforms.Compose([
        transforms.Resize((300, 300)),  # Tamaño esperado para SSD
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)  # Añadir batch dimension

    # Realizar predicción
    with torch.no_grad():
        predictions = ssd_model(img_tensor)

    # Procesar y mostrar resultados
    st.write("Predicciones del modelo SSD:")
    for i, (box, label, score) in enumerate(zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores'])):
        if score > 0.5:  # Filtrar por un umbral de confianza
            st.write(f"Objeto {i+1}: Clase {label} con {score:.2f} de confianza")
            # Opcional: puedes dibujar los bounding boxes sobre la imagen.