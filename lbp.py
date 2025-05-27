import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm

# Rutas y etiquetas
carpeta_base = 'data/entrenamiento'
categorias = ['pedestrian', 'no pedestrian']
etiquetas = {'pedestrian': 1, 'no pedestrian': 0}

# Parámetros LBP
radius = 1
n_points = 8 * radius
method = 'uniform'

def extraer_lbp(imagen_gris):
    lbp = local_binary_pattern(imagen_gris, n_points, radius, method=method)
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalizar histograma
    return hist

# Cargar datos y extraer características
X = []
y = []

for categoria in categorias:
    carpeta = os.path.join(carpeta_base, categoria)
    for nombre_archivo in tqdm(os.listdir(carpeta), desc=f"Procesando {categoria}"):
        if nombre_archivo.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            ruta = os.path.join(carpeta, nombre_archivo)
            imagen = cv2.imread(ruta)
            if imagen is None:
                continue
            imagen = cv2.resize(imagen, (64, 128))
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

            caracteristicas = extraer_lbp(imagen_gris)
            X.append(caracteristicas)
            y.append(etiquetas[categoria])

X = np.array(X)
y = np.array(y)

print("Datos cargados. Total muestras:", len(X))

# Separar en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar SVM
clf = LinearSVC(max_iter=10000)
clf.fit(X_train, y_train)

# Evaluar en validación
y_pred = clf.predict(X_val)
print("Reporte de clasificación:\n", classification_report(y_val, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_val, y_pred))

# Guardar modelo entrenado
os.makedirs('modelos', exist_ok=True)
with open('modelos/svm_entrenado_lbp.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Modelo SVM entrenado con LBP guardado en modelos/svm_entrenado_lbp.pkl")
