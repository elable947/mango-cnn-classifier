import os
import shutil
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)
processed_dir = Path("data/processed")
classes = ["Tipo_1", "Tipo_2", "Tipo_3", "Tipo_4", "Tipo_5"]

data_list = []
for cls in classes:
    cls_dir = processed_dir / cls
    if cls_dir.exists():
        for img_name in os.listdir(cls_dir):
            data_list.append({"path": str(cls_dir / img_name), "label": cls})

paths = [d["path"] for d in data_list]
labels = [d["label"] for d in data_list]

X_train, X_temp, y_train, y_temp = train_test_split(paths, labels, test_size=0.3, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Total imágenes: {len(paths)}")
print(f"Train: {len(X_train)} (70%)")
print(f"Validation: {len(X_val)} (15%)")
print(f"Test: {len(X_test)} (15%)")

splits_dict = {
    "train": [{"path": p, "label": l} for p, l in zip(X_train, y_train)],
    "val": [{"path": p, "label": l} for p, l in zip(X_val, y_val)],
    "test": [{"path": p, "label": l} for p, l in zip(X_test, y_test)]
}

with open("data/splits.json", "w", encoding="utf-8") as f:
    json.dump(splits_dict, f, indent=4)
print("Splits guardados en data/splits.json")

base_data_dir = Path("data")
for split_name in ["train", "val", "test"]:
    split_dir = base_data_dir / split_name
    if split_dir.exists():
        shutil.rmtree(split_dir)
    for item in splits_dict[split_name]:
        dest_dir = split_dir / item["label"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["path"], dest_dir / Path(item["path"]).name)
print("Imágenes distribuidas en carpetas train/, val/, test/.")

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
    split_dir = base_data_dir / split
    if not split_dir.exists(): continue
    for root, _, files in os.walk(split_dir):
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
