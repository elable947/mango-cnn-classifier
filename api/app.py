import json
import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf

# ── Configuración ──────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
IMG_SIZE = (224, 224)

CLASS_LABELS = ["Tipo_1", "Tipo_2", "Tipo_3", "Tipo_4", "Tipo_5"]
EXPORTABLE_MAP = {
    "Tipo_1": True,
    "Tipo_2": True,
    "Tipo_3": True,
    "Tipo_4": False,
    "Tipo_5": False,
}

# ── Carga de modelos ───────────────────────────────────────────────────────
models = {}
preprocess_fns = {}

# ResNet50 (Transfer Learning)
resnet_path = os.path.join(MODEL_DIR, "resnet50_mango.keras")
if os.path.exists(resnet_path):
    models["resnet50"] = tf.keras.models.load_model(resnet_path)
    preprocess_fns["resnet50"] = tf.keras.applications.resnet50.preprocess_input
    print("Modelo cargado: resnet50_mango.keras (preprocess_input)")

# CNN propia
cnn_path = os.path.join(MODEL_DIR, "cnn_propia.keras")
if os.path.exists(cnn_path):
    models["cnn_propia"] = tf.keras.models.load_model(cnn_path)
    preprocess_fns["cnn_propia"] = lambda x: x / 255.0
    print("Modelo cargado: cnn_propia.keras (rescale 1/255)")

# ── Métricas ───────────────────────────────────────────────────────────────
metrics_path = os.path.join(MODEL_DIR, "metrics.json")
with open(metrics_path, "r", encoding="utf-8") as f:
    all_metrics = json.load(f)


# ── Helpers ────────────────────────────────────────────────────────────────

def preparar_imagen(file_storage, model_name):
    """Preprocesa una imagen con el pipeline del modelo indicado."""
    img = Image.open(file_storage).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_fns[model_name](img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def inferir_con_modelo(file_storage, model_name):
    """Predice con un modelo específico."""
    img_array = preparar_imagen(file_storage, model_name)
    pred = models[model_name].predict(img_array, verbose=0)
    idx = int(np.argmax(pred))
    probabilidad = round(float(pred[0][idx]) * 100, 2)
    tipo = CLASS_LABELS[idx]
    return {
        "tipo": tipo,
        "exportable": EXPORTABLE_MAP[tipo],
        "probabilidad": probabilidad,
    }


# ── Flask App ──────────────────────────────────────────────────────────────

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/models", methods=["GET"])
def get_models_info():
    """Devuelve info de ambos modelos: nombres y métricas."""
    return jsonify({
        "models": list(models.keys()),
        "metrics": all_metrics,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        return jsonify({
            "error": f"Formato no soportado: {ext}. Usá: {', '.join(allowed_ext)}"
        }), 400

    try:
        resultados = {}
        for name in models:
            resultados[name] = inferir_con_modelo(file, name)
            file.seek(0)  # rebobinar para el siguiente modelo
        return jsonify(resultados)
    except Exception as e:
        return jsonify({"error": f"Error en la inferencia: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "modelos_cargados": list(models.keys()),
        "clases": CLASS_LABELS,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
