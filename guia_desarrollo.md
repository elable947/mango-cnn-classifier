# Guía de Desarrollo — Proyecto Clasificación de Mangos con CNN

## Tecnologías base

| Herramienta | Versión / Detalle |
|---|---|
| Python | 3.10.4 (gestionado con `uv`) |
| uv | Inicializar con `uv init --python 3.10.4` |
| Git + GitHub | Trabajo por ramas, Pull Requests |
| Polars | DataFrames (en vez de pandas) |
| Marimo | Notebooks reactivos (en vez de Jupyter) |
| Docker (opcional) | Para despliegue contenedorizado |
| Linux sin GUI | Servidor de despliegue por consola |

---

## Roles del equipo (3 integrantes)

Cada integrante tiene responsabilidades en múltiples fases.

| Rol | Sigla | Integrante |
|---|---|---|
| **Data Scientist / Deep Learning Lead** | DS | |
| **Backend / DevOps Engineer** | BE | |
| **Frontend Engineer / QA** | FE | |

---

## Flujo de trabajo Git

### Estructura de ramas

```
main ──── integración final, solo merge via PR
  ├─ develop ──── rama de integración diaria
  │    ├─ ds/eda                    ← Data Scientist
  │    ├─ ds/preprocessing          ← Data Scientist
  │    ├─ ds/cnn-propia             ← Data Scientist
  │    ├─ ds/transfer-learning      ← Data Scientist
  │    ├─ ds/evaluacion             ← Data Scientist
  │    ├─ be/api                    ← Backend
  │    ├─ be/docker                 ← Backend
  │    ├─ fe/frontend               ← Frontend
  │    └─ fe/qa                     ← Frontend
  └─ docs                           ← documentación compartida
```

### Reglas

1. **Nadie hace push directo a `main` ni a `develop`**.
2. Cada tarea se trabaja en su rama (ej. `ds/eda`).
3. Completada la tarea → Pull Request a `develop`.
4. El **otro integrante revisa el código** antes de aprobar el PR.
5. Al final de cada fase → PR de `develop` a `main` (revisión grupal).
6. Commits en español o inglés, descriptivos: `git commit -m "eda: consolidate 10 folders into 5 types"`.

### Setup inicial del repo

```bash
# 1. Clonar (o ya estamos en la carpeta)
cd D:\Projects\mangow

# 2. Inicializar con uv (Python 3.10.4)
uv init --python 3.10.4

# 3. Crear rama develop y subir
git checkout -b develop
git push -u origin develop

# 4. Cada integrante clona y crea su primera rama
git checkout -b ds/eda        # Data Scientist
git checkout -b be/api        # Backend
git checkout -b fe/frontend   # Frontend
```

---

## Fase 1 — Exploración y depuración del dataset (15%)

### Data Scientist

| # | Paso | Detalle |
|---|---|---|
| 1.1 | Descargar dataset de Google Drive | URL: <https://drive.google.com/file/d/1ax5pETzFT-OW7bNl7B5dvw-55K8C9RnW> |
| 1.2 | Descomprimir en `data/raw/` | Mantener estructura original de 10 carpetas |
| 1.3 | Crear notebook `notebooks/01_eda.py` | |
| 1.4 | Inspeccionar carpetas duplicadas | "EXPORTABLE" vs "EXPORTAR", "NO EXPORTABLE" vs "NO EXPORTAR" |
| 1.5 | Verificar manualmente muestras | Revisar ~10 imágenes de cada carpeta duplicada para confirmar que son la misma clase |
| 1.6 | Consolidar en 5 tipos | Fusionar en `data/processed/` con estructura: `Tipo_1/`, `Tipo_2/`, `Tipo_3/`, `Tipo_4/`, `Tipo_5/` |
| 1.7 | Analizar balance de clases | Calcular conteo, % del dataset, graficar barras |
| 1.8 | Analizar resolución y calidad | Histograma de dimensiones, detectar imágenes corruptas |
| 1.9 | Documentar hallazgos | En `docs/hallazgos_dataset.md` |

### Backend & Frontend

| # | Paso | Detalle |
|---|---|---|
| — | Revisar PR del DS | Code review del notebook y documentación |
| — | Familiarizarse con el dataset | Entender los 5 tipos y su relación con exportabilidad |

### Entregable
- `notebooks/01_eda.py` con todo el EDA
- `data/processed/` con las 5 carpetas consolidadas
- `docs/hallazgos_dataset.md`

---

## Fase 2 — Preprocesamiento (10%)

### Data Scientist

| # | Paso | Detalle |
|---|---|---|
| 2.1 | Crear notebook `notebooks/02_preprocessing.py` | |
| 2.2 | Definir tamaño de entrada | Ej. 224×224 (compatible con ResNet50) |
| 2.3 | Resize y normalización | Escalar píxeles a [0,1] |
| 2.4 | Split train/validation/test | 70% / 15% / 15% estratificado (mantener proporción por clase) |
| 2.5 | Data Augmentation | Rotaciones (±20°), flip horizontal, zoom, brillo, desplazamiento |
| 2.6 | Crear generadores/tensores | `data/train/`, `data/val/`, `data/test/` o usando `ImageDataGenerator` / `torchvision.transforms` |
| 2.7 | Guardar splits en archivos | `data/splits.json` con rutas y etiquetas |

### Backend & Frontend

| # | Paso | Detalle |
|---|---|---|
| — | Revisar PR | Verificar que los splits sean correctos |
| — | Actualizar docs | Si se definieron rutas de interés para la API |

### Entregable
- `notebooks/02_preprocessing.py`
- Datos augmentados y divididos
- `data/splits.json`

---

## Fase 3 — CNN propia (20%)

### Data Scientist

| # | Paso | Detalle |
|---|---|---|
| 3.1 | Crear notebook `notebooks/03_cnn_propia.py` | |
| 3.2 | Definir arquitectura | Conv2D + MaxPooling + Dropout + Dense + Softmax (5 salidas) |
| 3.3 | Compilar modelo | optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] |
| 3.4 | Entrenar | epochs=50-100, batch_size=32, EarlyStopping, ReduceLROnPlateau |
| 3.5 | Guardar mejor modelo | `models/cnn_propia.h5` o `.keras` (solo el de menor validation loss) |
| 3.6 | Graficar curvas de entrenamiento | loss y accuracy de train vs val |
| 3.7 | Analizar overfitting | Si train_loss ≪ val_loss, aumentar Dropout / regularización L2 |
| 3.8 | Subir notebook y modelo a la rama | |

### Backend & Frontend

| # | Paso | Detalle |
|---|---|---|
| — | Revisar PR | Verificar métricas, revisar curvas de entrenamiento |

### Entregable
- `notebooks/03_cnn_propia.py`
- `models/cnn_propia.keras`

---

## Fase 4 — Transfer Learning (15%)

### Data Scientist

| # | Paso | Detalle |
|---|---|---|
| 4.1 | Crear notebook `notebooks/04_transfer_learning.py` | |
| 4.2 | Cargar ResNet50 preentrenado (ImageNet) | `ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))` |
| 4.3 | Congelar capas base | `base_model.trainable = False` |
| 4.4 | Añadir cabezal propio | GlobalAveragePooling2D + Dense(256, ReLU) + Dropout + Dense(5, Softmax) |
| 4.5 | Fine-tuning (opcional) | Descongelar últimas capas y re-entrenar con lr muy bajo |
| 4.6 | Entrenar y guardar | `models/resnet50_mango.h5` |
| 4.7 | Comparar con CNN propia | Tabla comparativa de accuracy, F1-score y tamaño del modelo |

### Entregable
- `notebooks/04_transfer_learning.py`
- `models/resnet50_mango.keras`
- Tabla comparativa CNN propia vs Transfer Learning

---

## Fase 5 — Evaluación (10%)

### Data Scientist

| # | Paso | Detalle |
|---|---|---|
| 5.1 | Cargar mejor modelo (el que tenga mejor accuracy en test) | |
| 5.2 | Predecir sobre `data/test/` | |
| 5.3 | Calcular métricas por tipo | Accuracy, precision, recall, F1-score para cada uno de los 5 tipos |
| 5.4 | Matriz de confusión | Graficar heatmap de 5×5 |
| 5.5 | Métrica exportable vs no exportable | Agrupar T1+T2+T3 (exportable) vs T4+T5 (no exportable). Calcular accuracy, F1, matriz 2×2 |
| 5.6 | Calcular probabilidad promedio | Confianza media por clase |
| 5.7 | Guardar resultados | `models/metrics.json` con todas las métricas |

### Backend & Frontend

| # | Paso | Detalle |
|---|---|---|
| — | Revisar PR | Confirmar que las métricas sean consistentes |

### Entregable
- `notebooks/05_evaluacion.py`
- `models/metrics.json`
- Matrices de confusión (5×5 y 2×2)

---

## Fase 6 — API + Despliegue (20%)

### Backend (lidera) + Data Scientist (apoya)

| # | Paso | Detalle | Rol |
|---|---|---|---|
| 6.1 | Crear `api/app.py` con Flask | Endpoint `POST /predict` que recibe imagen + devuelve JSON | BE |
| 6.2 | Cargar modelo entrenado | Al iniciar la app, cargar el `.keras` con `load_model()` | BE |
| 6.3 | Preprocesar imagen recibida | Redimensionar a 224×224, normalizar, expandir dimensión batch | BE |
| 6.4 | Inferencia y formateo | `model.predict()` → argmax para tipo, max_prob * 100 para probabilidad, derivar exportable | BE |
| 6.5 | Entregar el mejor modelo al Backend | DS entrega el modelo final, explica formato esperado | DS |
| 6.6 | Formato JSON de respuesta | `{"tipo": "Tipo 2 (80%)", "exportable": true, "probabilidad": 95.4}` | BE |
| 6.7 | Manejo de errores | 400 si no hay imagen, 500 si falla inferencia | BE |
| 6.8 | Crear `requirements.txt` | flask, tensorflow, numpy, pillow, gunicorn | BE |
| 6.9 | Probar localmente | `python api/app.py` y probar con curl/Postman | BE |
| 6.10 | Crear `Dockerfile` | Imagen con Python 3.10.4, copiar api/ y models/, instalar dependencias, EXPOSE 5000, CMD con gunicorn | BE |
| 6.11 | Probar contenedor local | `docker build -t mango-api .` y `docker run -d -p 5000:5000 mango-api` | BE |
| 6.12 | O crear servicio systemd | `mango-api.service` que ejecute gunicorn con el venv | BE |
| 6.13 | Documentar comandos de despliegue | En `docs/despliegue.md`: comandos exactos para iniciar, detener, reiniciar, logs | BE |
| 6.14 | Probar endpoint completo | curl -X POST -F "image=@mango.jpg" http://localhost:5000/predict | BE |

### Entregable
- `api/app.py`
- `api/requirements.txt` (o `pyproject.toml` actualizado)
- `Dockerfile` (si se elige Docker)
- `docs/despliegue.md`

---

## Fase 7 — Frontend + QA (10%)

### Frontend (lidera) + Backend (apoya en integración)

| # | Paso | Detalle | Rol |
|---|---|---|---|
| 7.1 | Crear `frontend/index.html` | HTML5 + CSS + JS vanilla (o Angular) | FE |
| 7.2 | Input de imagen | `<input type="file" accept="image/*">` + preview | FE |
| 7.3 | Botón "Clasificar" | Al hacer click: `fetch('/predict', { method: 'POST', body: formData })` | FE |
| 7.4 | Mostrar resultado | Card con: tipo, exportable (✅ verde / ❌ rojo), probabilidad | FE |
| 7.5 | Manejo de errores | Mostrar mensaje si no hay imagen, si falla conexión o si hay error 400/500 | FE |
| 7.6 | Hacer que Flask sirva el frontend | En `app.py`: `app.static_folder = '../frontend'` o ruta separada | BE |
| 7.7 | Pruebas funcionales (QA) | Probar flujo completo: cargar imagen → predecir → mostrar resultado | FE |
| 7.8 | Pruebas de borde | Imagen corrupta, formato no soportado, archivo vacío, servidor caído | FE |
| 7.9 | Pruebas de integración | Frontend + API + modelo real, verificar que coinciden predicciones | FE + BE |

### Entregable
- `frontend/index.html` (y CSS/JS si aplica)
- `frontend/` completo servido por Flask

---

## Dependencias Python (pyproject.toml / requirements.txt)

```
numpy>=2.2.6
tensorflow>=2.13
flask>=3.0
gunicorn>=21.2
pillow>=10.0
matplotlib>=3.7
scikit-learn>=1.3
polars>=1.0
marimo>=0.23
seaborn>=0.13
```

Agregar con uv:
```bash
uv add tensorflow flask gunicorn pillow matplotlib scikit-learn polars marimo seaborn
```

### Notebooks con marimo

Los notebooks son archivos `.py` que se ejecutan con marimo:

```bash
# Crear/editar notebook
marimo edit notebooks/01_eda.py

# Ejecutar como app
marimo run notebooks/01_eda.py
```

---

## Referencia: proyecto COVID de ejemplo (`ejemplo_api/`)

El profesor proporcionó un proyecto completo de clasificación de radiografías (COVID vs Normal vs Neumonía) como guía. Está en `ejemplo_api/` (**no se sube a Git** por su peso, ~336 MB).

### Lo que podemos reutilizar directamente

| Componente | Archivo | Qué aprovechar |
|---|---|---|
| API Flask | `API_COVID.py` | Estructura base: cargar modelo, preparar imagen, endpoint `/predict`, manejo de errores 400/500, archivo temporal |
| Frontend | `Index.html` | HTML + Bootstrap 4 + JS vanilla, `FormData`, `fetch()`, preview de imagen, badges de colores por clase |
| Modelo | `modelo_cnn_covid.h5` | Referencia de cómo se guarda/carga un modelo entrenado con TensorFlow |

### Diferencias clave con nuestro proyecto

| Aspecto | Ejemplo COVID | Nuestro proyecto |
|---|---|---|
| Clases | 3 (Covid, Normal, Viral Pneumonia) | 5 tipos + flag exportable booleano |
| Formato modelo | `.h5` | `.keras` (más moderno) |
| CORS | `flask_cors` | Lo mismo, o usar el built-in de Flask |
| JSON respuesta | `clase`, `confianza_porcentaje`, `probabilidades` | `tipo`, `exportable`, `probabilidad` |
| Umbral | 85% para certeza alta | No requerido, pero se puede agregar |

### Notas importantes del ejemplo

1. **`flask_cors`** no está en nuestro `pyproject.toml`. Si se necesita habilitar CORS, agregarlo con `uv add flask-cors`.
2. El modelo se carga **una sola vez al iniciar** la app (`tf.keras.models.load_model()` a nivel módulo), no en cada request.
3. El preprocesamiento es: `Image.open → resize(224,224) → /255.0 → expand_dims`. Mismo pipeline que usaremos.
4. El archivo temporal se guarda, se procesa y se elimina en `finally` — buena práctica para no llenar el disco.
5. El frontend usa `fetch()` directo contra `http://127.0.0.1:5000/predict`. Para despliegue real se debe parametrizar o usar ruta relativa.

---

## Preguntas de análisis (para el informe final)

Estas preguntas deben responderse en la documentación:

1. ¿Qué problemas encontraron en la organización original de las carpetas y cómo los resolvieron?
2. ¿Qué ventajas ofrece una CNN frente a un modelo tradicional de ML para esta tarea?
3. ¿Por qué es importante el Data Augmentation dado el desbalance entre los 5 tipos?
4. ¿Cómo se detectó y manejó el desbalance de clases?
5. ¿Qué señales de overfitting se observaron y cómo se mitigaron?
6. ¿Qué rol cumple la función Softmax en la capa de salida?
7. Si el modelo confunde Tipo 3 (exportable) con Tipo 4 (no exportable), ¿qué impacto tiene ese error vs el error inverso? ¿Qué métrica priorizar?
8. ¿Qué mejoras para distintas condiciones de luz o fondo?
9. ¿Ventajas y desventajas de Docker vs systemd?

---

## Checklist final de entregables

| Archivo / Carpeta | Responsable |
|---|---|
| `notebooks/01_eda.py` | DS |
| `notebooks/02_preprocessing.py` | DS |
| `notebooks/03_cnn_propia.py` | DS |
| `notebooks/04_transfer_learning.py` | DS |
| `notebooks/05_evaluacion.py` | DS |
| `models/cnn_propia.keras` | DS |
| `models/resnet50_mango.keras` | DS |
| `models/metrics.json` | DS |
| `api/app.py` | BE |
| `Dockerfile` | BE |
| `docs/despliegue.md` | BE |
| `frontend/index.html` | FE |
| `docs/hallazgos_dataset.md` | DS |
| `guia_desarrollo.md` | (este archivo) |
