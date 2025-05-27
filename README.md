# Tarea 2 - Procesamiento Digital de Imágenes 🧠📷

Este proyecto consiste en el desarrollo de un sistema de **detección de peatones** a partir de imágenes estáticas, utilizando técnicas de **extracción de características** (HOG y LBP) y clasificación con **SVM (Support Vector Machine)**.

## Estructura del repositorio

```
Tarea2PDI/
├── data/
│   ├── entrenamiento/           # Imágenes de entrenamiento incluye con y sin peatones
│   └── prueba/           # Imágenes prueba incluye con y sin peatones
├── modelos/
│   ├── svm_entrenado_hog.pkl    # Modelo SVM entrenado con HOG
│   └── svm_entrenado_lbp.pkl # Modelo SVM entrenado con LBP
├── detector_hog.py           # Interfaz gráfica para probar el modelo HOG
├── detector_lbp.py          # Interfaz gráfica para probar el modelo LBP
├── hog.py                   # Entrenamiento del modelo SVM con HOG
├── lbp.py                   # Entrenamiento del modelo SVM con LBP
└── README.md                
```

---

## Requisitos

Asegúrate de tener Python 3 y los siguientes paquetes instalados:

```bash
pip install numpy opencv-python scikit-learn scikit-image pillow tqdm
```

---

## Cómo entrenar los modelos

### Con HOG:

```bash
python hog.py
```

Genera el archivo `modelos/svm_entrenado_hog.pkl` con el clasificador entrenado usando características **HOG**.

### Con LBP:

```bash
python lbp.py
```

Genera el archivo `modelos/svm_entrenado_lbp.pkl` usando características **LBP**.

---

## Cómo usar la interfaz

Ambas interfaces permiten cargar una imagen desde tu PC y clasificarla como "peatón" o "no peatón".

### HOG:

```bash
python detector.py
```

### LBP:

```bash
python detector_lbp.py
```

---

## Resultados

| Modelo    | Precisión | Recall | Accuracy |
|-----------|-----------|--------|----------|
| HOG + SVM | 0.66      | 0.66   | 66%      |
| LBP + SVM | 0.60      | 0.59   | 59%      |

**Conclusión:** El modelo basado en HOG tuvo un mejor desempeño general, ya que captura mejor las formas y contornos característicos de los peatones.

---

## Mejoras propuestas

- Aplicar técnicas de aumento de datos para equilibrar el dataset.
- Ajustar hiperparámetros del SVM mediante GridSearchCV.
- Implementar normalización o reducción de dimensionalidad (PCA).
- Combinar características HOG y LBP para mejorar la discriminación.


