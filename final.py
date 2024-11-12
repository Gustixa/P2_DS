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
