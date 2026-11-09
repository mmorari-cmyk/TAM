import os
import sys

# Forzar backend PyTorch para Keras 3
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import keras
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_model.keras")

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

@st.cache_resource
def load_keras_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    return keras.saving.load_model(MODEL_PATH)

def preprocess_opencv_single_digit(cv_img):
    """
    OpenCV preprocessing for a SINGLE handwritten digit:
    1. Background detection and Otsu / adaptive background subtraction.
    2. Connected components analysis to isolate the main digit and merge its strokes.
    3. Normalization and centering by Center of Mass (moments) on a 28x28 canvas.
    """
    if len(cv_img.shape) == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_img

    # 1. Background detection and binarization
    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg_median = float(np.median(border_pixels))
    max_val = float(np.max(gray))
    bg_is_dark = (bg_median < 50) and (max_val - bg_median > 80)

    if bg_is_dark:
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        k_size = max(25, int(min(gray.shape[:2]) * 0.1) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        bg = cv2.medianBlur(bg, k_size)

        norm = cv2.divide(gray, bg, scale=255)
        p_paper = float(np.median(norm))
        p_dark = float(np.percentile(norm, 1))
        T = p_dark + 0.35 * (p_paper - p_dark)
        bin_img = (norm < T).astype(np.uint8) * 255

    # 2. Connected components to find the single main digit
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img)
    if num_labels <= 1:
        return None, bin_img

    total_pixels = bin_img.shape[0] * bin_img.shape[1]
    min_area = max(40, int(total_pixels * 0.0003))

    valid_indices = [
        i for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]

    if not valid_indices:
        valid_indices = [1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]

    # Find the largest stroke of the digit
    main_idx = max(valid_indices, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    mx, my, mw, mh, _ = stats[main_idx]

    # Merge nearby disconnected strokes belonging to the same digit (e.g. crossbars in 4, 5, 7)
    x1, y1, x2, y2 = mx, my, mx + mw, my + mh
    pad = max(mw, mh) * 0.4
    for i in valid_indices:
        if i == main_idx:
            continue
        ix, iy, iw, ih, _ = stats[i]
        if not (ix > x2 + pad or ix + iw < x1 - pad or iy > y2 + pad or iy + ih < y1 - pad):
            x1 = min(x1, ix)
            y1 = min(y1, iy)
            x2 = max(x2, ix + iw)
            y2 = max(y2, iy + ih)

    w = x2 - x1
    h = y2 - y1

    # 3. Crop and resize maintaining aspect ratio into 20x20
    digit_crop = bin_img[y1:y2, x1:x2]
    dh, dw = digit_crop.shape
    if dh == 0 or dw == 0:
        return None, bin_img

    factor = 20.0 / max(dh, dw)
    new_w = max(1, int(round(dw * factor)))
    new_h = max(1, int(round(dh * factor)))
    resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    dilated = cv2.dilate(resized, np.ones((2, 2), np.uint8), iterations=1)

    # 4. Center by Center of Mass on a 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[(28 - new_h) // 2:(28 - new_h) // 2 + new_h, (28 - new_w) // 2:(28 - new_w) // 2 + new_w] = dilated

    M = cv2.moments(canvas)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        shift_x = 14.0 - cx
        shift_y = 14.0 - cy
        T_mat = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, T_mat, (28, 28), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    input_vector = canvas.astype(np.float32).reshape(1, 784)
    return {
        "box": (int(x1), int(y1), int(w), int(h)),
        "canvas": canvas,
        "vector": input_vector
    }, bin_img

# UI Header
st.title("🔢 Clasificador de Dígito MNIST")
st.markdown("Reconocimiento de **un único dígito** manuscrito mediante **OpenCV** y **Red Neuronal (Keras/PyTorch)**.")
st.caption("Desarrollado en pareja por **Jair Telpiz** y **Nathalia Mora** | Teoría de Aprendizaje de Máquinas (TAM)")

# Load model
try:
    model = load_keras_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Input mode selection
st.sidebar.header("Opciones de Entrada")
input_mode = st.sidebar.radio("Fuente de Imagen:", ["Subir Foto de Cuaderno", "Tomar Foto con Cámara"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Integrantes:**")
st.sidebar.markdown("• **Jair Telpiz**  \n• **Nathalia Mora**")
st.sidebar.caption("Teoría de Aprendizaje de Máquinas (TAM)")

uploaded_file = None
if input_mode == "Subir Foto de Cuaderno":
    uploaded_file = st.sidebar.file_uploader(
        "Elegí una imagen (JPG, PNG, JPEG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"]
    )
else:
    uploaded_file = st.sidebar.camera_input("Capturá el dígito con tu cámara")

if uploaded_file is not None:
    # Read image into OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if cv_img is None:
        st.error("No se pudo decodificar la imagen.")
        st.stop()

    with st.spinner("Procesando dígito con OpenCV y ejecutando inferencia..."):
        digit_item, bin_img = preprocess_opencv_single_digit(cv_img)

    if digit_item is None:
        st.warning("No se detectó un dígito claro en la imagen. Probá con mayor contraste o mejor iluminación.")
    else:
        # Run inference for the single digit
        pred_probs = model.predict(digit_item["vector"], verbose=0)[0]
        pred_digit = int(np.argmax(pred_probs))
        conf = float(pred_probs[pred_digit]) * 100

        # Annotate image
        annotated_img = cv_img.copy()
        x, y, w, h = digit_item["box"]
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 220, 0), 3)
        cv2.putText(
            annotated_img,
            f"{pred_digit} ({conf:.1f}%)",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 220, 0),
            2
        )
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # Main result metric
        st.metric(
            label="Dígito Detectado",
            value=f"{pred_digit}",
            delta=f"{conf:.1f}% certeza"
        )

        # Side by side visualizer
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Imagen Original con Detección")
            st.image(annotated_rgb, use_container_width=True)

        with col2:
            st.subheader("2. Visión Normalizada (28x28 Keras)")
            st.image(digit_item["canvas"], width=220, caption=f"Entrada a la red: dígito {pred_digit}")

        # Probability distribution bar chart
        st.subheader("Distribución de Probabilidades (0 al 9)")
        df_probs = pd.DataFrame({
            "Dígito": [str(d) for d in range(10)],
            "Probabilidad (%)": pred_probs * 100
        })
        st.bar_chart(df_probs.set_index("Dígito"))
else:
    st.info("Subí una foto de un único dígito en tu cuaderno o capturalo con la cámara.")
