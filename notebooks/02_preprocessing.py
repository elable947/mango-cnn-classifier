import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")

@app.cell
def __():
    import os
    import shutil
    import json
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    return ImageDataGenerator, Path, json, os, shutil, tf, train_test_split


@app.cell
def __(Path, os):
    # 2.2 Definir tamaño de entrada
    IMG_SIZE = (224, 224)
    processed_dir = Path("data/processed")
    classes = ["Tipo_1", "Tipo_2", "Tipo_3", "Tipo_4", "Tipo_5"]
    
    # Recopilar rutas y etiquetas
    data_list = []
    for cls in classes:
        cls_dir = processed_dir / cls
        if cls_dir.exists():
            for img_name in os.listdir(cls_dir):
                data_list.append({"path": str(cls_dir / img_name), "label": cls})
    return IMG_SIZE, classes, data_list, processed_dir


@app.cell
def __(data_list, train_test_split):
    # 2.4 Split train/validation/test (70% / 15% / 15%)
    paths = [d["path"] for d in data_list]
    labels = [d["label"] for d in data_list]

    # Primero separamos 70% train y 30% temp (que será val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(paths, labels, test_size=0.3, stratify=labels, random_state=42)

    # Luego separamos el 30% en dos mitades (15% val, 15% test)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    print(f"Total imágenes: {len(paths)}")
    print(f"Train: {len(X_train)} (70%)")
    print(f"Validation: {len(X_val)} (15%)")
    print(f"Test: {len(X_test)} (15%)")
    return X_temp, X_test, X_train, X_val, labels, paths, y_temp, y_test, y_train, y_val


@app.cell
def __(Path, X_test, X_train, X_val, json, y_test, y_train, y_val):
    # 2.7 Guardar splits en data/splits.json
    splits_dict = {
        "train": [{"path": p, "label": l} for p, l in zip(X_train, y_train)],
        "val": [{"path": p, "label": l} for p, l in zip(X_val, y_val)],
        "test": [{"path": p, "label": l} for p, l in zip(X_test, y_test)]
    }

    with open("data/splits.json", "w", encoding="utf-8") as f:
        json.dump(splits_dict, f, indent=4)
    print("Splits guardados en data/splits.json")
    return splits_dict,


@app.cell
def __(Path, os, shutil, splits_dict):
    # Copiar imágenes a las carpetas respectivas data/train, data/val, data/test
    # para usar flow_from_directory
    base_data_dir = Path("data")
    for split_name in ["train", "val", "test"]:
        split_dir = base_data_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir) # Limpiar si existía
        for item in splits_dict[split_name]:
            dest_dir = split_dir / item["label"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item["path"], dest_dir / Path(item["path"]).name)
    print("Imágenes distribuidas en carpetas train/, val/, test/.")
    return base_data_dir, dest_dir, item, split_dir, split_name


@app.cell
def __(Path, base_data_dir, os):
    import hashlib
    def get_file_hash(filepath):
        hasher = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                buf = f.read()
                hasher.update(buf)
            return hasher.hexdigest()
        except Exception:
            return None

    print("\nRealizando auditoría de Data Leakage...")
    seen_hashes = {}
    duplicates = 0
    for split in ["train", "val", "test"]:
        split_dir_audit = base_data_dir / split
        if not split_dir_audit.exists(): continue
        for root, _, files in os.walk(split_dir_audit):
            for filename in files:
                filepath = Path(root) / filename
                if filepath.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}: continue
                f_hash = get_file_hash(filepath)
                if not f_hash: continue
                if f_hash in seen_hashes:
                    os.remove(filepath)
                    duplicates += 1
                else:
                    seen_hashes[f_hash] = filepath
    print(f"Auditoría completada. Fugas detectadas y eliminadas: {duplicates}")
    return duplicates, get_file_hash, seen_hashes, split_dir_audit


@app.cell
def __(IMG_SIZE, ImageDataGenerator):
    # 2.3 Resize y normalización [0,1]
    # 2.5 Data Augmentation
    # 2.6 Crear generadores/tensores
    train_datagen = ImageDataGenerator(
        rescale=1./255,           # Normalización
        rotation_range=20,        # Data Augmentation: Rotación
        width_shift_range=0.1,    # Desplazamiento
        height_shift_range=0.1,
        zoom_range=0.2,           # Zoom
        horizontal_flip=True,     # Flip
        fill_mode='nearest'
    )

    # Validation y Test solo necesitan rescale
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    batch_size = 32

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_test_datagen.flow_from_directory(
        'data/val',
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        'data/test',
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    return (
        batch_size,
        test_generator,
        train_datagen,
        train_generator,
        val_generator,
        val_test_datagen,
    )


if __name__ == "__main__":
    app.run()
