import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.feature import hog
from skimage import color
import pickle

# Parámetros HOG (deben ser iguales que en el entrenamiento)
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

# Cargar modelo SVM guardado
with open('modelos/svm_entrenado.pkl', 'rb') as f:
    clf = pickle.load(f)

def cargar_imagen():
    ruta = filedialog.askopenfilename(filetypes=[("Imagenes", "*.jpg *.jpeg *.png *.bmp")])
    if not ruta:
        return
    imagen = cv2.imread(ruta)
    if imagen is None:
        messagebox.showerror("Error", "No se pudo abrir la imagen")
        return

    # Mostrar imagen en la interfaz
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen_pil = Image.fromarray(imagen_rgb)
    imagen_pil = imagen_pil.resize((256, 256))  # ajustar tamaño visual
    imagen_tk = ImageTk.PhotoImage(imagen_pil)
    lbl_imagen.config(image=imagen_tk)
    lbl_imagen.image = imagen_tk

    # Procesar y predecir
    img_resized = cv2.resize(imagen, (64, 128))
    img_gray = color.rgb2gray(img_resized)
    caracteristicas = hog(img_gray, visualize=False, **hog_params).reshape(1, -1)
    prediccion = clf.predict(caracteristicas)[0]

    texto = "Peatón detectado" if prediccion == 1 else "No hay peatón"
    lbl_resultado.config(text=texto)

# Crear ventana principal
ventana = tk.Tk()
ventana.title("Detector de Peatones")

btn_cargar = tk.Button(ventana, text="Cargar imagen", command=cargar_imagen)
btn_cargar.pack(pady=10)

lbl_imagen = tk.Label(ventana)
lbl_imagen.pack()

lbl_resultado = tk.Label(ventana, text="", font=("Arial", 14))
lbl_resultado.pack(pady=10)

ventana.mainloop()