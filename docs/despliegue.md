# Guía de Despliegue — Mango CNN Classifier API

## Requisitos

- Docker 20+ instalado
- Al menos **8 GB de RAM** libres (~420 MB para ambos modelos + TensorFlow en CPU)
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

Ambos modelos tardan ~120s en cargar en CPU. Verificá con:

```bash
curl http://localhost:5000/health
```

Respuesta esperada:
```json
{"clases":["Tipo_1","Tipo_2","Tipo_3","Tipo_4","Tipo_5"],"modelos_cargados":["resnet50","cnn_propia"],"status":"ok"}
```

## Probar predicción

```bash
curl -X POST -F "image=@ruta/a/tu/mango.jpg" http://localhost:5000/predict
```

Ejemplo de respuesta (ambos modelos):
```json
{
  "resnet50": {"exportable": false, "probabilidad": 94.47, "tipo": "Tipo_4"},
  "cnn_propia": {"exportable": false, "probabilidad": 89.12, "tipo": "Tipo_4"}
}
```

## Ver métricas

```bash
curl http://localhost:5000/models
```

## Logs

```bash
docker logs mango-api         # una sola vez
docker logs -f mango-api       # seguir en tiempo real (Ctrl+C para salir)
```

## Ver contenedores

```bash
docker ps                      # solo corriendo
docker ps -a                   # todos (incluyendo detenidos)
```

## Detener

```bash
docker stop mango-api
```

## Iniciar (contenedor existente)

```bash
docker start mango-api
```

⚠️ Después de iniciar, esperar ~120s para que TensorFlow recargue los modelos.

## Borrar

```bash
docker rm -f mango-api
```

---

## Solución de problemas

| Error | Causa probable | Solución |
|---|---|---|
| `WORKER TIMEOUT` | Inferencia lenta en CPU | La predicción con 2 modelos tarda ~2-3 min, es normal |
| `address already in use` | Puerto 5000 ocupado | `docker rm -f mango-api` y reintentar |
| `No se encontró el modelo` | Modelos no están en `models/` | Ejecutar `git lfs pull` |
| La respuesta tarda >3 min | CPU sin GPU - 2 modelos | Normal. Con GPU tarda <1s |

---

## Stack

| Componente | Detalle |
|---|---|
| Base image | `python:3.10.4-slim` |
| Servidor WSGI | gunicorn (1 worker, timeout 600s) |
| Framework | Flask |
| ML | TensorFlow |
| Modelos | ResNet50 (Transfer Learning) + CNN Propia |
| Frontend | HTML5 + Bootstrap 4.6 + JS vanilla |
