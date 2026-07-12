import os
import shutil
import polars as pl
from pathlib import Path
from PIL import Image

raw_dir = Path("data/raw/Dataset_mangos")
folders = [f.name for f in raw_dir.iterdir() if f.is_dir()]
print("Carpetas originales encontradas:")
for f in sorted(folders):
    print(f" - {f} ({len(os.listdir(raw_dir / f))} imágenes)")

processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

mapping = {
    "Tipo_1": ["TIPO 1_ 100__EXPORTAR", "TIPO 1_100__EXPORTABLE"],
    "Tipo_2": ["TIPO 2_80__EXPORTABLE", "TIPO 2_80__EXPORTAR"],
    "Tipo_3": ["TIPO 3_60__EXPORTABLE", "TIPO 3_60__EXPORTAR"],
    "Tipo_4": ["TIPO 4_40__NO EXPORTABLE", "TIPO 4_40__NO EXPORTAR"],
    "Tipo_5": ["TIPO 5_20__NO EXPORTABLE", "TIPO 5_20__NO EXPORTAR"]
}

print("\nConsolidando imágenes...")
for target_class, source_folders in mapping.items():
    target_dir = processed_dir / target_class
    target_dir.mkdir(exist_ok=True)
    count = 0
    for src in source_folders:
        src_path = raw_dir / src
        if src_path.exists():
            for img_name in os.listdir(src_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    new_name = f"{src}_{img_name}".replace(" ", "_")
                    shutil.copy2(src_path / img_name, target_dir / new_name)
                    count += 1
    print(f"{target_class}: {count} imágenes consolidadas.")

print("\nAnalizando imágenes consolidadas...")
stats = []
corrupted = []
for cls_dir in processed_dir.iterdir():
    if cls_dir.is_dir():
        imgs = os.listdir(cls_dir)
        cls_name = cls_dir.name
        for img_name in imgs:
            img_path = cls_dir / img_name
            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    w, h = img.size
                    stats.append({"Clase": cls_name, "Ancho": w, "Alto": h})
            except Exception as e:
                print(f"Imagen corrupta: {img_path} - {e}")
                corrupted.append(img_path)

df_stats = pl.DataFrame(stats)
class_counts = df_stats.group_by("Clase").len().sort("Clase")
print("\nBalance de clases:")
print(class_counts)

print("\nEstadísticas de resolución:")
print(df_stats.select(pl.col("Ancho"), pl.col("Alto")).describe())
