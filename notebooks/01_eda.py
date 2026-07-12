import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import os
    import shutil
    import glob
    from pathlib import Path
    from PIL import Image
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    return Image, Path, glob, os, pl, plt, sns, shutil


@app.cell
def __(Path, os):
    # 1.4 Inspeccionar carpetas duplicadas
    raw_dir = Path("data/raw/Dataset_mangos")
    folders = [f.name for f in raw_dir.iterdir() if f.is_dir()]
    print("Carpetas originales encontradas:")
    for f in sorted(folders):
        print(f" - {f} ({len(os.listdir(raw_dir / f))} imágenes)")
    return folders, raw_dir


@app.cell
def __(Path, folders, os, raw_dir, shutil):
    # 1.6 Consolidar en 5 tipos
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    mapping = {
        "Tipo_1": ["TIPO 1_ 100__EXPORTAR", "TIPO 1_100__EXPORTABLE"],
        "Tipo_2": ["TIPO 2_80__EXPORTABLE", "TIPO 2_80__EXPORTAR"],
        "Tipo_3": ["TIPO 3_60__EXPORTABLE", "TIPO 3_60__EXPORTAR"],
        "Tipo_4": ["TIPO 4_40__NO EXPORTABLE", "TIPO 4_40__NO EXPORTAR"],
        "Tipo_5": ["TIPO 5_20__NO EXPORTABLE", "TIPO 5_20__NO EXPORTAR"]
    }
    
    print("Consolidando imágenes...")
    for target_class, source_folders in mapping.items():
        target_dir = processed_dir / target_class
        target_dir.mkdir(exist_ok=True)
        count = 0
        for src in source_folders:
            src_path = raw_dir / src
            if src_path.exists():
                for img_name in os.listdir(src_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # To avoid name collisions, prefix with folder name
                        prefix = "exp_" if "EXPORTAR" in src or "EXPORTABLE" in src else "no_exp_"
                        new_name = f"{src}_{img_name}".replace(" ", "_")
                        shutil.copy2(src_path / img_name, target_dir / new_name)
                        count += 1
        print(f"{target_class}: {count} imágenes consolidadas.")
    return count, mapping, processed_dir


@app.cell
def __(Image, Path, os, processed_dir):
    # 1.7 & 1.8 Analizar balance y calidad
    stats = []
    widths = []
    heights = []
    corrupted = []
    
    for cls_dir in processed_dir.iterdir():
        if cls_dir.is_dir():
            imgs = os.listdir(cls_dir)
            cls_name = cls_dir.name
            for img_name in imgs:
                img_path = cls_dir / img_name
                try:
                    with Image.open(img_path) as img:
                        img.verify() # Verify integrity
                    # Open again to get size as verify() might close it
                    with Image.open(img_path) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                        stats.append({"Clase": cls_name, "Ancho": w, "Alto": h})
                except Exception as e:
                    print(f"Imagen corrupta: {img_path} - {e}")
                    corrupted.append(img_path)
    return cls_dir, cls_name, corrupted, heights, img, img_name, img_path, imgs, stats, w, widths


@app.cell
def __(pl, stats):
    df_stats = pl.DataFrame(stats)
    class_counts = df_stats.group_by("Clase").len().sort("Clase")
    print("Balance de clases:")
    print(class_counts)
    return class_counts, df_stats


@app.cell
def __(class_counts, df_stats, plt, sns):
    plt.figure(figsize=(10, 5))
    sns.barplot(data=class_counts.to_pandas(), x="Clase", y="len", palette="viridis")
    plt.title("Balance de Clases (Consolidado)")
    plt.ylabel("Cantidad de Imágenes")
    plt.show()
    
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df_stats.to_pandas(), x="Ancho", y="Alto", hue="Clase", alpha=0.5)
    plt.title("Distribución de Resoluciones")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
