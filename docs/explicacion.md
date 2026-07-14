# Explicación Técnica — Mango CNN Classifier

> Proyecto: Clasificación de Mangos con CNN | Roles: Backend (Fase 6) + Frontend (Fase 7)
> Basado en `guia_desarrollo.md` y el ejemplo `API_COVID.py` del profesor.

---

## 1. Comando `netsh interface portproxy`

```powershell
netsh interface portproxy add v4tov4 listenport=5000 listenaddress=0.0.0.0 connectport=5000 connectaddress=172.29.130.59
```

### ¿Qué hace?

WSL2 corre en una máquina virtual Hyper-V con su **propia IP privada** (`172.29.130.59`). Esa IP no es accesible desde fuera del host Windows. Este comando crea un **proxy de puertos a nivel de kernel** en Windows que redirige:

```
Mundo exterior → Windows (0.0.0.0:5000) → WSL2 (172.29.130.59:5000)
```

### Desglose

| Parámetro | Significado |
|---|---|
| `listenport=5000` | Puerto en el que Windows escucha |
| `listenaddress=0.0.0.0` | Escuchar en todas las interfaces de red (WiFi, Ethernet, Tailscale) |
| `connectport=5000` | Puerto en WSL al que reenviar |
| `connectaddress=172.29.130.59` | IP interna de WSL2 |

Sin esto, `http://100.82.22.10:5000` (IP de Tailscale) no llega a Flask porque WSL está aislado.

---

## 2. Comunicación Frontend ↔ Backend

### Flujo completo de una predicción

```
┌──────────────┐     HTTP POST /predict      ┌──────────────┐
│  Navegador   │ ──────────────────────────> │  Flask API   │
│  (frontend)  │                              │  (backend)   │
│              │ <──────────────────────────  │              │
│ index.html   │     JSON response            │  app.py      │
└──────────────┘                              └──────┬───────┘
                                                     │
                                            model.predict()
                                                     │
                                              ┌──────▼───────┐
                                              │  TensorFlow  │
                                              │  ResNet50    │
                                              │  (165 MB)    │
                                              └──────────────┘
```

### Paso a paso

1. **Usuario** selecciona una imagen en `index.html` (input type="file")
2. **Frontend** muestra preview con `FileReader.readAsDataURL()`
3. **Usuario** hace click en "Clasificar"
4. **Frontend** crea `FormData` y envía la imagen vía `fetch("/predict", {method: "POST", body: formData})`
5. **Flask** recibe la request en `request.files["image"]` — es un objeto `FileStorage`
6. **Flask** preprocesa: `Image.open(file).resize(224,224)` → `preprocess_input()` → `expand_dims()`
7. **Flask** llama a `model.predict(img_array)` — TensorFlow ejecuta la inferencia
8. **Flask** procesa el resultado: `argmax` para el tipo, `max*100` para probabilidad, lookup en `EXPORTABLE_MAP`
9. **Flask** devuelve JSON: `{"tipo": "Tipo_4", "exportable": false, "probabilidad": 94.47}`
10. **Frontend** recibe el JSON, renderiza la card con badge exportable y barra de progreso

### ¿Por qué `fetch("/predict")` y no `fetch("http://IP:5000/predict")`?

Usamos **ruta relativa**. Cuando el navegador carga `index.html` desde `http://100.82.22.10:5000/`, la ruta `/predict` se resuelve automáticamente como `http://100.82.22.10:5000/predict`. Esto funciona tanto en:

| Entorno | URL base | `/predict` se resuelve a |
|---|---|---|
| Desarrollo (Flask directo) | `http://172.29.130.59:5000` | `http://172.29.130.59:5000/predict` |
| Tailscale + portproxy | `http://100.82.22.10:5000` | `http://100.82.22.10:5000/predict` |
| Docker | `http://localhost:5000` | `http://localhost:5000/predict` |
| Producción con nginx | `https://midominio.com` | `https://midominio.com/predict` |

No se hardcodea ninguna IP. El profesor lo pide así en la guía (ver nota 5 de la referencia COVID).

### ¿Cómo sirve Flask el frontend?

```python
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")
```

Flask configura `static_folder` apuntando a `frontend/`. Cuando el navegador pide `GET /`, Flask devuelve `index.html`. Los assets estáticos (CSS, JS, imágenes) también se sirven desde ahí. Esto cumple el paso **7.6** de la guía: *"Hacer que Flask sirva el frontend"*.

### Comunicación en Docker

Dentro del contenedor, la estructura es:

```
/app/
  api/app.py         ← Flask (puerto 5000)
  frontend/index.html ← Servido por Flask
  models/*.keras     ← Modelos
```

El `Dockerfile` copia `frontend/` junto con `api/` y `models/`. Gunicorn ejecuta `api.app:app`, que a su vez sirve el frontend. El puerto 5000 del contenedor se mapea al puerto 5000 del host con `-p 5000:5000`.

---

## 3. Backend — `api/app.py` (Fase 6)

### Arquitectura general

```
Cliente (navegador / curl)
       │
       ▼
   Flask (0.0.0.0:5000)
       │
       ├── GET  /          → frontend/index.html (HTML)
       ├── GET  /health    → {"status": "ok", "modelo": "...", "clases": [...]}
       └── POST /predict   → {"tipo": "...", "exportable": bool, "probabilidad": float}
                                  │
                                  ▼
                         TensorFlow + ResNet50
                         (modelo cargado UNA vez al iniciar)
```

### Carga del modelo (paso 6.2)

```python
model = tf.keras.models.load_model("models/resnet50_mango.keras")
```

El modelo se carga **a nivel módulo**, no dentro de cada request. Esto significa que se carga **una sola vez** cuando arranca Flask, igual que en el ejemplo `API_COVID.py` del profesor. Si se cargara en cada request, cada predicción tardaría ~90s adicionales.

### Selección de modelo (línea 11)

```python
MODEL_NAME = "resnet50_mango"   # Cambiar a "cnn_propia" para el otro modelo
```

El preprocesamiento se adapta automáticamente:

| Modelo | Preprocesamiento |
|---|---|
| `resnet50_mango` | `tf.keras.applications.resnet50.preprocess_input` (normalización específica de ImageNet) |
| `cnn_propia` | `img_array / 255.0` (reescalado simple a [0, 1]) |

Es **crítico** usar el mismo preprocesamiento que se usó durante el entrenamiento, o el modelo dará predicciones erróneas.

### Endpoint `POST /predict` (paso 6.1, 6.3, 6.4)

Flujo completo:

```
1. Validar request.files["image"]      → 400 si no hay imagen
2. Validar filename no vacío            → 400 si vacío
3. Validar extensión (.jpg/.png/.bmp)  → 400 si formato no soportado
4. Image.open(file).convert("RGB")     → asegurar 3 canales
5. img.resize((224, 224))              → tamaño fijo para el modelo
6. np.array(img, dtype=np.float32)     → convertir a array NumPy
7. preprocess_fn(img_array)            → normalizar según modelo
8. np.expand_dims(img_array, axis=0)   → agregar batch dimension (1, 224, 224, 3)
9. model.predict(img_array)            → inferencia
10. np.argmax(pred)                    → índice de la clase con mayor probabilidad
11. Exportable según mapa              → Tipo_1/2/3 = true, Tipo_4/5 = false
12. jsonify(resultado)                 → respuesta HTTP
```

### Formato de respuesta (paso 6.6)

```json
{
  "tipo": "Tipo_4",
  "exportable": false,
  "probabilidad": 94.47
}
```

### Mapa de exportación

| Clase | Exportable | Criterio original |
|---|---|---|
| Tipo_1 | ✅ true | 100% exportable |
| Tipo_2 | ✅ true | 80% exportable |
| Tipo_3 | ✅ true | 60% exportable |
| Tipo_4 | ❌ false | 40% no exportable |
| Tipo_5 | ❌ false | 20% no exportable |

### Manejo de errores (paso 6.7)

| Caso | Código HTTP | Respuesta |
|---|---|---|
| Sin imagen | 400 | `{"error": "No se envió ninguna imagen"}` |
| Nombre vacío | 400 | `{"error": "Nombre de archivo vacío"}` |
| Formato no soportado | 400 | `{"error": "Formato no soportado: .txt. Usá: .jpg, .jpeg, .png, .bmp"}` |
| Imagen corrupta | 500 | `{"error": "Error en la inferencia: ..."}` |
| Error interno | 500 | `{"error": "Error en la inferencia: ..."}` |

### Endpoint `GET /health`

Endpoint auxiliar para monitoreo. Devuelve metadatos sin tocar el modelo:

```json
{"clases": ["Tipo_1","Tipo_2","Tipo_3","Tipo_4","Tipo_5"], "modelo": "resnet50_mango", "status": "ok"}
```

### Endpoint `GET /`

Sirve el frontend. Flask configura `static_folder` a `../frontend` y devuelve `index.html` como página principal.

---

## 4. Frontend — `frontend/index.html` (Fase 7)

### Tecnologías

| Componente | Tecnología |
|---|---|
| Estructura | HTML5 semántico |
| Estilos | Bootstrap 4.6 CDN + CSS inline |
| Lógica | JavaScript vanilla (sin frameworks) |
| Comunicación | `fetch()` API nativa del navegador |

La guía pedía "HTML5 + CSS + JS vanilla (o Angular)". Se eligió vanilla JS siguiendo el ejemplo del profesor.

### Componentes de la UI

| Componente | Descripción | Paso |
|---|---|---|
| `<input type="file">` | Selector de imagen con filtro `.jpg,.jpeg,.png,.bmp` | 7.2 |
| `previewImage()` | `FileReader.readAsDataURL()` → `<img>` de previsualización | 7.2 |
| Botón "Clasificar" | Dispara `clasificar()` | 7.3 |
| Loader spinner | Bootstrap spinner + texto "Analizando imagen..." (importante: CPU tarda >60s) | Extra |
| Result card | Badge verde/rojo para exportable + barra de progreso para probabilidad | 7.4 |
| Alertas de error | Warning amarillo (sin imagen) o danger rojo (error servidor) | 7.5 |

### Flujo del frontend

```
1. Usuario selecciona archivo → previewImage()
2. Usuario hace click en "Clasificar" → clasificar()
3. Validar que haya archivo seleccionado
4. Mostrar loader, deshabilitar botón
5. new FormData() + append("image", file)
6. fetch("/predict", { method: "POST", body: formData })
7. Si !res.ok → extraer error del JSON y mostrar alerta roja
8. Si ok → renderizar card con tipo, badge exportable, barra %
9. finally → ocultar loader, habilitar botón
```

### Por qué `fetch("/predict")` sin dominio absoluto

Usa ruta relativa. En desarrollo apunta al mismo host (WSL:5000). En producción con nginx reverse proxy también funciona sin cambios.

---

## 5. Docker (Fase 6.10–6.11)

### ¿Por qué Docker?

La guía lo pide como opción de despliegue (alternativa a systemd). Ventajas:

- **Portabilidad**: mismo entorno en desarrollo y producción
- **Aislamiento**: dependencias no contaminan el sistema host
- **Reproducibilidad**: cualquier persona con Docker puede levantar la API

### Dockerfile explicado

```dockerfile
FROM python:3.10.4-slim          # Imagen base ligera (la guía pide Python 3.10.4)
WORKDIR /app                      # Directorio de trabajo

RUN apt-get install libgl1...     # Dependencias del sistema para Pillow
COPY api/requirements.txt .       # Copiar dependencias Python
RUN pip install -r requirements.txt
COPY api/app.py api/              # Código de la API
COPY frontend/ frontend/          # Frontend estático
COPY models/ models/              # Modelos .keras

ENV TF_ENABLE_ONEDNN_OPTS=0       # Evitar conflictos oneDNN + gunicorn en CPU
EXPOSE 5000
CMD ["gunicorn", ...]             # WSGI server para producción
```

### ¿Por qué gunicorn y no `flask run`?

El servidor de desarrollo de Flask (`app.run()`) **no es para producción**. Gunicorn es un servidor WSGI probado en producción que:

- Maneja workers separados (1 en nuestro caso por el peso del modelo)
- Timeout configurable (600s por la lentitud de CPU)
- `--max-requests 1`: recicla el worker después de cada request (evita memory leaks de TF)
- `--max-requests-jitter 10`: añade aleatoriedad al reciclaje

### Gunicorn + TensorFlow en CPU

TensorFlow y gunicorn tienen problemas conocidos al forkear procesos. Solución aplicada:

| Problema | Solución |
|---|---|
| oneDNN + fork causa crashes | `TF_ENABLE_ONEDNN_OPTS=0` |
| Worker timeout en carga | `--timeout 600` |
| 2 workers = 2× RAM | `--workers 1` |

### Comandos Docker principales

```bash
docker build -t mango-api .                            # Construir imagen
docker run -d -p 5000:5000 --name mango-api mango-api  # Iniciar (crear nuevo)
docker start mango-api                                 # Iniciar (contenedor existente)
docker ps                                              # Ver contenedores corriendo
docker ps -a                                           # Ver todos (incluyendo detenidos)
docker logs -f mango-api                               # Ver logs en vivo
docker stop mango-api                                  # Detener
docker rm -f mango-api                                 # Borrar
```

---

## 6. Pruebas realizadas (Fase 7.7–7.9)

### Pruebas funcionales (7.7)

| Prueba | Comando | Resultado esperado | Resultado |
|---|---|---|---|
| Imagen válida (Tipo 4) | `curl -F "image=@mango.jpg" /predict` | `{"tipo":"Tipo_4","exportable":false,"probabilidad":94.47}` | ✅ |
| Health check | `curl /health` | `{"status":"ok",...}` | ✅ |
| Frontend HTML | `curl /` | HTTP 200, `<!DOCTYPE html>` | ✅ |

### Pruebas de borde (7.8)

| Prueba | Comando | Resultado esperado | Resultado |
|---|---|---|---|
| Sin archivo | `curl -X POST /predict` | 400 `"No se envió ninguna imagen"` | ✅ |
| Archivo corrupto | `echo "basura" > fake.jpg` → enviar | 500 `"cannot identify image file"` | ✅ |
| Formato no soportado | enviar `.txt` | 400 `"Formato no soportado"` | ✅ |
| Servidor caído | `pkill -f api/app.py` → request | Error de conexión (frontend muestra alerta) | ✅ |

### Pruebas de integración (7.9)

Flujo completo verificado: Frontend → API → Modelo → Respuesta:
- `GET /` → HTML del frontend (HTTP 200)
- `POST /predict` con imagen real → JSON correcto con tipo, exportable, probabilidad
- El frontend parsea correctamente el JSON y renderiza la card de resultado

---

## 7. Resumen de archivos del entregable

| Archivo | Fase | Descripción |
|---|---|---|
| `api/app.py` | 6.1–6.7 | API Flask con 3 endpoints |
| `api/requirements.txt` | 6.8 | Dependencias Python para producción |
| `Dockerfile` | 6.10 | Contenedor Docker con gunicorn |
| `docs/despliegue.md` | 6.13 | Guía de comandos Docker |
| `frontend/index.html` | 7.1–7.5 | Interfaz web HTML5 + Bootstrap + JS |
| `docs/explicacion.md` | — | Este documento |
| `docs/presentacion.tex` | — | Presentación Beamer LaTeX |
