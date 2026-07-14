import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf

# ── Configuración ──────────────────────────────────────────────────────────
# Cambiá MODEL_NAME para elegir el modelo:
#   "resnet50_mango" → Transfer Learning (ResNet50)
#   "cnn_propia"     → CNN propia desde cero
MODEL_NAME = "resnet50_mango"

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
IMG_SIZE = (224, 224)

# Tipos de mango y su condición de exportación
CLASS_LABELS = ["Tipo_1", "Tipo_2", "Tipo_3", "Tipo_4", "Tipo_5"]
EXPORTABLE_MAP = {
    "Tipo_1": True,
    "Tipo_2": True,
    "Tipo_3": True,
    "Tipo_4": False,
    "Tipo_5": False,
}

# ── Carga del modelo ───────────────────────────────────────────────────────
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"No se encontró el modelo en {MODEL_PATH}. "
        f"Asegurate de que el archivo existe en models/."
    )

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Modelo cargado: {MODEL_NAME}.keras")

# Determinar el preprocesamiento según el modelo
if "resnet" in MODEL_NAME.lower():
    preprocess_fn = tf.keras.applications.resnet50.preprocess_input
    print("Preprocesamiento: resnet50.preprocess_input")
else:
    # CNN propia: solo rescale a [0, 1]
    preprocess_fn = lambda x: x / 255.0
    print("Preprocesamiento: rescale 1/255")


# ── Helpers ────────────────────────────────────────────────────────────────

def preparar_imagen(file_storage):
    """Preprocesa una imagen subida al endpoint.

    Args:
        file_storage: Objeto FileStorage de Flask (request.files["image"]).

    Returns:
        np.ndarray con forma (1, 224, 224, 3) lista para inferencia.
    """
    img = Image.open(file_storage).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_fn(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def inferir(img_array):
    """Ejecuta inferencia sobre la imagen preprocesada.

    Args:
        img_array: np.ndarray con forma (1, 224, 224, 3).

    Returns:
        dict con 'tipo', 'exportable' y 'probabilidad'.
    """
    pred = model.predict(img_array, verbose=0)
    idx = int(np.argmax(pred))
    probabilidad = round(float(pred[0][idx]) * 100, 2)
    tipo = CLASS_LABELS[idx]
    exportable = EXPORTABLE_MAP[tipo]

    return {
        "tipo": tipo,
        "exportable": exportable,
        "probabilidad": probabilidad,
    }


# ── Flask App ──────────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Validar que se envió una imagen
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    # Validar extensión
    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify({
            "error": f"Formato no soportado: {ext}. Usá: {', '.join(allowed_ext)}"
        }), 400

    try:
        img_array = preparar_imagen(file)
        resultado = inferir(img_array)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": f"Error en la inferencia: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "modelo": MODEL_NAME,
        "clases": CLASS_LABELS,
    })


# ── Entrada principal ──────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
