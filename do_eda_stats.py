import polars as pl
from pathlib import Path
from PIL import Image
import os
import json

processed_dir = Path("data/processed")
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
                    w, h = img.size
                    stats.append({"Clase": cls_name, "Ancho": w, "Alto": h})
            except Exception as e:
                corrupted.append(str(img_path))

df_stats = pl.DataFrame(stats)
class_counts = df_stats.group_by("Clase").len().sort("Clase")
desc = df_stats.select(pl.col("Ancho"), pl.col("Alto")).describe()

with open("stats.json", "w", encoding="utf-8") as f:
    json.dump({"counts": class_counts.to_dicts(), "desc": desc.to_dicts(), "corrupted": corrupted}, f)
