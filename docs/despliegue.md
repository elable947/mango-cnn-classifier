# Guía de Despliegue — Mango CNN Classifier API

## Requisitos

- Docker 20+ instalado
- Al menos **8 GB de RAM** libres (ResNet50 + TensorFlow en CPU)
- Puertos: **5000** disponible

---

## Build

Construir la imagen desde la raíz del proyecto:

```bash
docker build -t mango-api .
```

## Iniciar

```bash
docker run -d -p 5000:5000 --name mango-api mango-api
```

El modelo tarda ~90s en cargar en CPU. Verificá con:

```bash
curl http://localhost:5000/health
```

Respuesta esperada:
```json
{"clases":["Tipo_1","Tipo_2","Tipo_3","Tipo_4","Tipo_5"],"modelo":"resnet50_mango","status":"ok"}
```

## Probar predicción

```bash
curl -X POST -F "image=@ruta/a/tu/mango.jpg" http://localhost:5000/predict
```

Ejemplo de respuesta:
```json
{"exportable": true, "probabilidad": 95.42, "tipo": "Tipo_1"}
```

## Logs

```bash
docker logs mango-api        # una sola vez
docker logs -f mango-api      # seguir en tiempo real (Ctrl+C para salir)
```

## Detener

```bash
docker stop mango-api
```

## Reiniciar

```bash
docker restart mango-api
```

⚠️ Después de reiniciar, esperar ~90s para que TensorFlow recargue el modelo.

## Borrar

```bash
docker rm -f mango-api
```

---

## Cambiar de modelo

Editar `api/app.py`, línea 11:

```python
MODEL_NAME = "cnn_propia"    # CNN propia (255 MB, rescale 1/255)
# MODEL_NAME = "resnet50_mango"  # ResNet50 Transfer Learning (165 MB, preprocess_input)
```

Reconstruir y relanzar:

```bash
docker rm -f mango-api
docker build -t mango-api .
docker run -d -p 5000:5000 --name mango-api mango-api
```

---

## Solución de problemas

| Error | Causa probable | Solución |
|---|---|---|
| `WORKER TIMEOUT` | Inferencia lenta en CPU | Aumentar RAM disponible o usar GPU |
| `address already in use` | Puerto 5000 ocupado | `docker rm -f mango-api` y reintentar |
| `No se encontró el modelo` | Modelo no está en `models/` | Ejecutar `git lfs pull` |
| La respuesta tarda >2 min | CPU sin GPU | Normal. Con GPU tarda <1s |

---

## Stack

| Componente | Detalle |
|---|---|
| Base image | `python:3.10.4-slim` |
| Servidor WSGI | gunicorn (1 worker, timeout 600s) |
| Framework | Flask |
| ML | TensorFlow + ResNet50 |
| Modelo por defecto | `models/resnet50_mango.keras` |
