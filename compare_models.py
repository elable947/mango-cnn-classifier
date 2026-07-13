import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, f1_score

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def get_model_size_mb(path):
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 * 1024)

# 1. Evaluar CNN Propia
print("Evaluando CNN Propia...")
cnn_model_path = 'models/cnn_propia.keras'
cnn_size = get_model_size_mb(cnn_model_path)
cnn_model = load_model(cnn_model_path)

cnn_test_datagen = ImageDataGenerator(rescale=1./255)
cnn_test_generator = cnn_test_datagen.flow_from_directory(
    'data/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

cnn_preds_probs = cnn_model.predict(cnn_test_generator)
cnn_preds = np.argmax(cnn_preds_probs, axis=1)
cnn_true = cnn_test_generator.classes

cnn_acc = accuracy_score(cnn_true, cnn_preds)
cnn_f1 = f1_score(cnn_true, cnn_preds, average='weighted')
print(f"CNN Propia - Acc: {cnn_acc:.4f}, F1: {cnn_f1:.4f}")

# 2. Evaluar ResNet50
print("Evaluando ResNet50 (Transfer Learning)...")
resnet_model_path = 'models/resnet50_mango.keras'
resnet_size = get_model_size_mb(resnet_model_path)
resnet_model = load_model(resnet_model_path)

resnet_test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
resnet_test_generator = resnet_test_datagen.flow_from_directory(
    'data/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

resnet_preds_probs = resnet_model.predict(resnet_test_generator)
resnet_preds = np.argmax(resnet_preds_probs, axis=1)
resnet_true = resnet_test_generator.classes

resnet_acc = accuracy_score(resnet_true, resnet_preds)
resnet_f1 = f1_score(resnet_true, resnet_preds, average='weighted')
print(f"ResNet50 - Acc: {resnet_acc:.4f}, F1: {resnet_f1:.4f}")

# 3. Escribir tabla comparativa
md_content = f"""# Comparación de Modelos (Fase 4)

| Modelo | Accuracy | F1-Score (Weighted) | Tamaño en Disco |
|--------|----------|---------------------|-----------------|
| **CNN Propia** | {cnn_acc:.4f} | {cnn_f1:.4f} | {cnn_size:.2f} MB |
| **ResNet50 (Transfer Learning)** | {resnet_acc:.4f} | {resnet_f1:.4f} | {resnet_size:.2f} MB |

## Conclusión Breve
* El modelo **ResNet50** tiene un tamaño de {resnet_size:.2f} MB frente a los {cnn_size:.2f} MB de la CNN Propia.
* La Accuracy de la CNN Propia es de {cnn_acc:.4f} frente a {resnet_acc:.4f} de ResNet50.
* El F1-score es de {cnn_f1:.4f} para la CNN Propia y de {resnet_f1:.4f} para ResNet50.
"""

os.makedirs('docs', exist_ok=True)
with open('docs/comparacion_modelos.md', 'w', encoding='utf-8') as f:
    f.write(md_content)

print("\nTabla comparativa guardada en docs/comparacion_modelos.md")
