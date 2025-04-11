
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io
from PIL import Image

st.set_page_config(page_title="Segmentación de Imágenes con K-Means", layout="wide")

st.title("🎨 Segmentación de Imágenes con K-Means")

st.markdown("""
Esta aplicación permite segmentar cualquier imagen en distintos colores utilizando el algoritmo **K-Means Clustering**.
""")

# Subida de imagen
uploaded_file = st.file_uploader("🔼 Sube una imagen (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image = np.array(image)

    st.subheader("📷 Imagen Original")
    st.image(image, use_column_width=True)

    n_clusters = st.slider("🎯 Selecciona el número de segmentos (K)", min_value=2, max_value=10, value=4)

    # Mostrar botón para segmentar
    if st.button("Segmentar Imagen"):
        st.info("Segmentando...")

        # Preprocesamiento
        img_data = image.reshape((-1, 3))
        img_data = np.float32(img_data)

        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(img_data)
        labels = kmeans.labels_
        centers = np.uint8(kmeans.cluster_centers_)

        # Reconstrucción
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(image.shape)

        st.subheader("🖼️ Imagen Segmentada")
        st.image(segmented_image, use_column_width=True)

        # Colores de los clusters
        st.subheader("🎨 Colores Dominantes")
        cluster_colors = np.reshape(centers, (1, n_clusters, 3))
        st.image(cluster_colors, width=300)
