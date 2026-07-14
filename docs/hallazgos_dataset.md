# Hallazgos del Dataset

## 1. Duplicidad e Inconsistencia de Carpetas
Al revisar el dataset original, se encontraron 10 carpetas con una mezcla del tipo de calidad del mango y su estado de exportabilidad. Existían carpetas duplicadas que representaban la misma clase pero con sufijos diferentes (por ejemplo, "EXPORTABLE" vs "EXPORTAR" y "NO EXPORTABLE" vs "NO EXPORTAR").

Se verificó manualmente una muestra de estas carpetas y se confirmó que corresponden al mismo tipo de calidad. Las carpetas fueron consolidadas en 5 clases principales:
- **Tipo_1 (100% Calidad, Exportable):** Fusión de "TIPO 1_ 100__EXPORTAR" y "TIPO 1_100__EXPORTABLE"
- **Tipo_2 (80% Calidad, Exportable):** Fusión de "TIPO 2_80__EXPORTABLE" y "TIPO 2_80__EXPORTAR"
- **Tipo_3 (60% Calidad, Exportable):** Fusión de "TIPO 3_60__EXPORTABLE" y "TIPO 3_60__EXPORTAR"
- **Tipo_4 (40% Calidad, No Exportable):** Fusión de "TIPO 4_40__NO EXPORTABLE" y "TIPO 4_40__NO EXPORTAR"
- **Tipo_5 (20% Calidad, No Exportable):** Fusión de "TIPO 5_20__NO EXPORTABLE" y "TIPO 5_20__NO EXPORTAR"

## 2. Balance de Clases
El dataset consta de **2,039 imágenes** en total. La distribución tras la consolidación es la siguiente:
- **Tipo_1**: 355 imágenes (17.4%)
- **Tipo_2**: 288 imágenes (14.1%)
- **Tipo_3**: 507 imágenes (24.9%)
- **Tipo_4**: 428 imágenes (21.0%)
- **Tipo_5**: 461 imágenes (22.6%)

Existe un ligero desbalance, siendo el **Tipo 2** la clase minoritaria (288 imágenes) y el **Tipo 3** la mayoritaria (507 imágenes). Será necesario aplicar técnicas de Data Augmentation para equilibrar el conjunto de datos durante la fase de entrenamiento y evitar que el modelo se sesgue hacia las clases mayoritarias.

## 3. Resolución y Calidad de las Imágenes
Se analizó la integridad y las dimensiones de todas las imágenes:
- **Resolución promedio:** ~140 x 152 píxeles
- **Resolución mínima:** 94 x 126 píxeles
- **Resolución máxima:** 168 x 168 píxeles
- **Imágenes corruptas:** 0 (Se verificó la integridad de todos los archivos y ninguna imagen presentó daños que impidan su lectura).

Dado que las resoluciones varían, durante la fase de preprocesamiento se deberá estandarizar el tamaño de entrada de la red convolucional (por ejemplo a 224x224 píxeles si se usa ResNet50 en la fase de Transfer Learning) realizando un _resize_ a las imágenes.
