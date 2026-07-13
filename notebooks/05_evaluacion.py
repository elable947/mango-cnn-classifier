import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

@app.cell
def __():
    import os
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    
    os.makedirs('docs', exist_ok=True)
    return (
        ImageDataGenerator,
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        json,
        load_model,
        np,
        os,
        plt,
        precision_score,
        recall_score,
        sns,
        tf,
    )

@app.cell
def __(ImageDataGenerator, load_model, tf):
    # 5.1 Cargar el mejor modelo (ResNet50)
    print("Cargando modelo ResNet50...")
    model = load_model('models/resnet50_mango.keras')

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    # 5.2 Predecir sobre data/test/
    test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return BATCH_SIZE, IMG_SIZE, model, test_datagen, test_generator

@app.cell
def __(model, np, test_generator):
    # Obtener predicciones
    print("Realizando predicciones...")
    preds_probs = model.predict(test_generator)
    preds = np.argmax(preds_probs, axis=1)
    true_labels = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    
    return class_labels, preds, preds_probs, true_labels

@app.cell
def __(
    class_labels,
    confusion_matrix,
    f1_score,
    plt,
    precision_score,
    preds,
    recall_score,
    sns,
    true_labels,
):
    # 5.3 Calcular métricas por tipo y 5.4 Matriz de confusión 5x5
    metrics_by_type = {}
    
    # Calcular métricas globales (macro)
    metrics_by_type["macro_precision"] = precision_score(true_labels, preds, average='macro')
    metrics_by_type["macro_recall"] = recall_score(true_labels, preds, average='macro')
    metrics_by_type["macro_f1"] = f1_score(true_labels, preds, average='macro')
    
    # Calcular por cada clase
    precision_arr = precision_score(true_labels, preds, average=None)
    recall_arr = recall_score(true_labels, preds, average=None)
    f1_arr = f1_score(true_labels, preds, average=None)
    
    for _i, _label in enumerate(class_labels):
        metrics_by_type[_label] = {
            "precision": float(precision_arr[_i]),
            "recall": float(recall_arr[_i]),
            "f1_score": float(f1_arr[_i])
        }
    
    # Matriz de confusión 5x5
    cm_5x5 = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_5x5, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Matriz de Confusión 5x5 (Tipos de Mango)')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig('docs/matriz_confusion_5x5.png')
    
    print("Métricas por tipo calculadas y matriz 5x5 guardada.")
    return cm_5x5, f1_arr, metrics_by_type, precision_arr, recall_arr

@app.cell
def __(
    accuracy_score,
    confusion_matrix,
    f1_score,
    plt,
    preds,
    sns,
    true_labels,
):
    # 5.5 Métrica exportable vs no exportable
    # T1, T2, T3 -> Exportable (1)
    # T4, T5 -> No Exportable (0)
    # Asumimos que los índices 0, 1, 2 son T1, T2, T3 (esto depende de class_indices, que está ordenado alfabéticamente)
    
    def map_to_exportable(label_idx):
        return 1 if label_idx in [0, 1, 2] else 0

    true_exp = [map_to_exportable(l) for l in true_labels]
    preds_exp = [map_to_exportable(p) for p in preds]

    acc_exp = accuracy_score(true_exp, preds_exp)
    f1_exp = f1_score(true_exp, preds_exp, pos_label=1)
    
    metrics_exportable = {
        "accuracy": float(acc_exp),
        "f1_score": float(f1_exp)
    }
    
    cm_2x2 = confusion_matrix(true_exp, preds_exp)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_2x2, annot=True, fmt='d', cmap='Greens', xticklabels=['No Exportable', 'Exportable'], yticklabels=['No Exportable', 'Exportable'])
    plt.title('Matriz de Confusión 2x2 (Exportable vs No Exportable)')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig('docs/matriz_confusion_2x2.png')
    
    print(f"Métricas Exportabilidad - Acc: {acc_exp:.4f}, F1: {f1_exp:.4f}")
    return acc_exp, cm_2x2, f1_exp, map_to_exportable, metrics_exportable, preds_exp, true_exp

@app.cell
def __(class_labels, np, preds_probs):
    # 5.6 Calcular probabilidad promedio (Confianza media por clase)
    avg_confidence = {}
    
    # preds_probs tiene shape (N, 5). Tomamos el máximo de cada fila como la "confianza"
    max_probs = np.max(preds_probs, axis=1)
    predicted_classes = np.argmax(preds_probs, axis=1)
    
    for _i, _label in enumerate(class_labels):
        # Filtrar las instancias donde el modelo predijo esta clase
        class_probs = max_probs[predicted_classes == _i]
        if len(class_probs) > 0:
            avg_confidence[_label] = float(np.mean(class_probs))
        else:
            avg_confidence[_label] = 0.0
            
    print("Confianza promedio por clase calculada:", avg_confidence)
    return avg_confidence, class_probs, max_probs, predicted_classes

@app.cell
def __(avg_confidence, json, metrics_by_type, metrics_exportable):
    # 5.7 Guardar resultados en models/metrics.json
    final_metrics = {
        "metrics_by_type": metrics_by_type,
        "metrics_exportable": metrics_exportable,
        "avg_confidence": avg_confidence
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
    print("Métricas guardadas exitosamente en models/metrics.json")
    return f, final_metrics

if __name__ == "__main__":
    app.run()
