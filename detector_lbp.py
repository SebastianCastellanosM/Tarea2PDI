import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage import color
import pickle

# Parámetros LBP
radius = 1
n_points = 8 * radius
method = 'uniform'

# Cargar modelo SVM entrenado con LBP
with open('modelos/svm_entrenado_lbp.pkl', 'rb') as f:
    clf = pickle.load(f)

def extraer_caracteristicas_lbp(imagen):
    imagen = cv2.resize(imagen, (64, 128))
    imagen_gris = color.rgb2gray(imagen)

    lbp = local_binary_pattern(imagen_gris, n_points, radius, method)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.reshape(1, -1)

def cargar_imagen():
    ruta = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
    if not ruta:
        return
    imagen = cv2.imread(ruta)
    if imagen is None:
        messagebox.showerror("Error", "No se pudo abrir la imagen")
        return

    # Mostrar imagen en la interfaz
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen_pil = Image.fromarray(imagen_rgb)
    imagen_pil = imagen_pil.resize((256, 256))
    imagen_tk = ImageTk.PhotoImage(imagen_pil)
    lbl_imagen.config(image=imagen_tk)
    lbl_imagen.image = imagen_tk

    # Extraer características y predecir
    caracteristicas = extraer_caracteristicas_lbp(imagen)
    prediccion = clf.predict(caracteristicas)[0]

    texto = "Peatón detectado" if prediccion == 1 else "No hay peatón"
    lbl_resultado.config(text=texto)

# Crear ventana principal
ventana = tk.Tk()
ventana.title("Detector SVM con LBP")

btn_cargar = tk.Button(ventana, text="Cargar imagen", command=cargar_imagen)
btn_cargar.pack(pady=10)

lbl_imagen = tk.Label(ventana)
lbl_imagen.pack()

lbl_resultado = tk.Label(ventana, text="", font=("Arial", 14))
lbl_resultado.pack(pady=10)

ventana.mainloop()
