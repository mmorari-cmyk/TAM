#!/home/jair/venv_modernas/bin/python
import os
import sys

# Forzar backend PyTorch para Keras 3
os.environ["KERAS_BACKEND"] = "torch"

import cv2
import keras
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_model.keras")

class DigitClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector MNIST - Pruebas con Cuaderno")
        self.root.geometry("720x650")
        self.root.minsize(680, 600)

        # Cargar modelo
        try:
            self.model = keras.saving.load_model(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Error al cargar modelo", f"No se pudo cargar '{MODEL_PATH}':\n{e}")
            sys.exit(1)

        self._crear_interfaz()

    def _crear_interfaz(self):
        # Header
        header = ttk.Frame(self.root, padding=12)
        header.pack(fill=tk.X)
        
        ttk.Label(
            header,
            text="Prueba de Red Neuronal MNIST con Fotos de Cuaderno",
            font=("Helvetica", 15, "bold")
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="OpenCV binariza, recorta y centra el dígito en 28x28 tal como espera la red.",
            font=("Helvetica", 10),
            foreground="#555555"
        ).pack(anchor="w", pady=(2, 0))

        # Botón selección
        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(fill=tk.X)
        ttk.Button(
            btn_frame,
            text="📂 Seleccionar Foto de Cuaderno...",
            command=self.cargar_imagen
        ).pack(side=tk.LEFT)

        self.lbl_path = ttk.Label(btn_frame, text="Ningún archivo seleccionado", foreground="#777777")
        self.lbl_path.pack(side=tk.LEFT, padx=12)

        # Contenedor de visualización
        vis_frame = ttk.Frame(self.root, padding=10)
        vis_frame.pack(fill=tk.BOTH, expand=True)

        # Panel izquierdo: Imagen Original
        left_box = ttk.LabelFrame(vis_frame, text=" 1. Imagen Original / Recortada ", padding=10)
        left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.lbl_orig = ttk.Label(left_box, text="Cargá una foto\npara comenzar", anchor="center")
        self.lbl_orig.pack(fill=tk.BOTH, expand=True)

        # Panel derecho: Visión OpenCV (28x28)
        right_box = ttk.LabelFrame(vis_frame, text=" 2. Lo que 've' la Red Neuronal (28x28 OpenCV) ", padding=10)
        right_box.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.lbl_processed = ttk.Label(right_box, text="Acá verás el dígito procesado\n(blanco sobre negro)", anchor="center")
        self.lbl_processed.pack(fill=tk.BOTH, expand=True)

        # Panel inferior: Resultados
        res_frame = ttk.LabelFrame(self.root, text=" Resultado de la Inferencia ", padding=12)
        res_frame.pack(fill=tk.X, padx=15, pady=10)

        self.lbl_result = ttk.Label(
            res_frame,
            text="Predicción: -",
            font=("Helvetica", 18, "bold"),
            foreground="#0066cc"
        )
        self.lbl_result.pack(anchor="center")

        self.lbl_conf = ttk.Label(
            res_frame,
            text="Esperando imagen...",
            font=("Helvetica", 11)
        )
        self.lbl_conf.pack(anchor="center", pady=(4, 6))

    def preprocesar_con_opencv(self, cv_img):
        """
        Segmentación y normalización OpenCV para un ÚNICO dígito:
        1. Detección automática de tipo de fondo y binarización.
        2. Aislamiento del trazo principal del dígito y agrupación de trazos cercanos.
        3. Centrado por centro de masa (moments) a 28x28.
        """
        if len(cv_img.shape) == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_img

        # 1. Detectar tipo de fondo
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

        # 2. Componentes conectados para aislar el único dígito
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

        # Encontrar el componente principal del dígito
        main_idx = max(valid_indices, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        mx, my, mw, mh, _ = stats[main_idx]

        # Agrupar trazos cercanos del mismo dígito (ej: barras en 4, 5, 7)
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

        # 3. Recortar y escalar manteniendo relación de aspecto a 20x20
        digit_crop = bin_img[y1:y2, x1:x2]
        dh, dw = digit_crop.shape
        if dh == 0 or dw == 0:
            return None, bin_img

        factor = 20.0 / max(dh, dw)
        new_w = max(1, int(round(dw * factor)))
        new_h = max(1, int(round(dh * factor)))
        resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        dilated = cv2.dilate(resized, np.ones((2, 2), np.uint8), iterations=1)

        # 4. Centrado por centro de masa en 28x28
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

    def cargar_imagen(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar foto de cuaderno",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos los archivos", "*.*")]
        )
        if not file_path:
            return

        self.lbl_path.config(text=os.path.basename(file_path))

        cv_img = cv2.imread(file_path)
        if cv_img is None:
            messagebox.showerror("Error", "No se pudo leer la imagen seleccionada.")
            return

        digit_item, _ = self.preprocesar_con_opencv(cv_img)

        if digit_item is None:
            self.lbl_result.config(text="No se detectó un dígito claro")
            self.lbl_conf.config(text="Intentá con otra imagen más nítida")
            return

        # Inferencia con Keras para el único dígito encontrado
        pred = self.model.predict(digit_item["vector"], verbose=0)[0]
        d = int(np.argmax(pred))
        conf = float(pred[d]) * 100

        annotated_img = cv_img.copy()
        x, y, w, h = digit_item["box"]
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(
            annotated_img,
            f"{d} ({conf:.1f}%)",
            (x, max(20, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 190, 0),
            2,
            cv2.LINE_AA
        )

        self.lbl_result.config(text=f"Dígito detectado:  {d}")
        self.lbl_conf.config(text=f"Confianza: {conf:.1f}%")

        # Renderizar Imagen con Bounding Box en panel izquierdo
        h_img, w_img = annotated_img.shape[:2]
        max_size = 280
        scale = min(max_size / w_img, max_size / h_img)
        preview_orig = cv2.resize(annotated_img, (int(w_img * scale), int(h_img * scale)))
        preview_orig_rgb = cv2.cvtColor(preview_orig, cv2.COLOR_BGR2RGB)
        im_orig = Image.fromarray(preview_orig_rgb)
        tk_orig = ImageTk.PhotoImage(im_orig)
        self.lbl_orig.config(image=tk_orig, text="")
        self.lbl_orig.image = tk_orig

        # Renderizar dígito 28x28 en panel derecho
        canvas = digit_item["canvas"]
        target_size = 180
        preview_proc = cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        im_proc = Image.fromarray(preview_proc)
        tk_proc = ImageTk.PhotoImage(im_proc)
        self.lbl_processed.config(image=tk_proc, text="")
        self.lbl_processed.image = tk_proc


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitClassifierApp(root)
    root.mainloop()
