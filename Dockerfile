FROM python:3.10.4-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para Pillow y OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias Python
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la API, frontend y modelo
COPY api/app.py api/
COPY frontend/ frontend/
COPY models/ models/

# Puerto expuesto
EXPOSE 5000

# Desactivar oneDNN para evitar conflictos con gunicorn en CPU
ENV TF_ENABLE_ONEDNN_OPTS=0

# Comando de inicio con gunicorn (producción)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "600", "--max-requests", "1", "--max-requests-jitter", "10", "api.app:app"]
