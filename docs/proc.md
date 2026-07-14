# Procedimiento — Día de la Presentación

> Cómo levantar todo desde cero el día que defendés el proyecto.
> Solo necesitás tu laptop con Zed (SSH) y la PC remota encendida.

---

## Resumen rápido

```bash
# 1. SSH a la PC remota (ya lo tenés en Zed)
# 2. En WSL:
cd ~/mango-cnn-classifier
docker rm -f mango-api 2>/dev/null
docker build -t mango-api .
docker run -d -p 5000:5000 --name mango-api mango-api

# 3. Esperar ~90s (carga del modelo en CPU)
curl localhost:5000/health   # Confirmar que responde

# 4. Verificar la IP de WSL:
hostname -I

# 5. En PowerShell (Windows, admin) ACTUALIZAR el portproxy:
netsh interface portproxy reset
netsh interface portproxy add v4tov4 listenport=5000 listenaddress=0.0.0.0 connectport=5000 connectaddress=<IP_DE_WSL>

# 5. Verificar IP de Tailscale:
tailscale ip -4

# 6. En tu laptop, abrí:
http://<IP_TAILSCALE>:5000
```

---

## Paso a paso detallado

### 1. Conectarte por SSH

Desde Zed en tu laptop ya tenés la conexión SSH configurada a la PC remota. Abrí la terminal de Zed.

### 2. Levantar Docker

```bash
# Por si quedó algo corriendo
docker rm -f mango-api 2>/dev/null

# Construir la imagen (solo si hubo cambios en el código)
docker build -t mango-api .

# Iniciar el contenedor
docker run -d -p 5000:5000 --name mango-api mango-api
```

### 3. Verificar que la API está viva

Esperá ~90 segundos (TensorFlow carga el modelo ResNet50 de 165 MB en CPU) y luego:

```bash
# Ver logs en tiempo real (Ctrl+C para salir sin matar el contenedor)
docker logs -f mango-api

# O verificá directamente
curl localhost:5000/health
```

Respuesta esperada:
```json
{"clases":["Tipo_1","Tipo_2","Tipo_3","Tipo_4","Tipo_5"],"modelo":"resnet50_mango","status":"ok"}
```

### 4. Actualizar el portproxy de Windows

La IP de WSL2 **cambia cada vez que se reinicia WSL**. El portproxy anterior apunta a una IP vieja.

En WSL, obtené la IP actual:

```bash
hostname -I
```

Te va a dar algo como `172.29.130.59`. En **PowerShell como administrador** en la PC remota:

```powershell
# Borrar reglas viejas
netsh interface portproxy reset

# Crear la nueva regla con la IP actual de WSL
netsh interface portproxy add v4tov4 listenport=5000 listenaddress=0.0.0.0 connectport=5000 connectaddress=172.29.XXX.XXX
```

> **Nota:** Si no reiniciaste la PC ni WSL desde la última vez, la IP no cambió y podés saltar este paso. Para estar seguro, ejecutalo igual.

### 5. Abrir en tu laptop

Verificá la IP de Tailscale (por si cambió):

```bash
tailscale ip -4
```

Luego en tu laptop abrí:

```
http://<IP_TAILSCALE>:5000
```

Ejemplo: `http://100.82.22.10:5000`

---

## Prueba rápida en vivo

Para demostrar que funciona, tené lista una imagen de mango. Desde WSL:

```bash
# Tipo 4 (no exportable) — respuesta en ~60s
curl -X POST -F "image=@Dataset_mangos/TIPO 4_40__NO EXPORTABLE/20230515_181028.jpg" http://localhost:5000/predict
```

O subila desde el frontend en el navegador.

---

## Troubleshooting

| Problema | Causa | Solución |
|---|---|---|
| `address already in use` | Puerto 5000 ocupado | `docker rm -f mango-api` y reintentar |
| `Cannot connect` en el navegador | Portproxy desactualizado o IP de Tailscale cambió | Revisar `tailscale ip -4` y `hostname -I`, actualizar ambos |
| `WORKER TIMEOUT` en logs | Docker sin RAM suficiente | Asegurar que la PC tiene >8 GB libres |
| El navegador carga pero no responde | Modelo todavía cargando | Esperar ~90s, revisar `docker logs` |
| `Permission denied` con docker | Usuario no en grupo docker | `sudo usermod -aG docker $USER` + re-login |

---

## ¿Qué persiste y qué no?

| Elemento | ¿Persiste al apagar? | ¿Persiste al reiniciar? |
|---|---|---|
| Código (`~/mango-cnn-classifier`) | ✅ Sí (disco) | ✅ Sí |
| Imagen Docker (`mango-api`) | ✅ Sí (disco) | ✅ Sí |
| Modelos `.keras` | ✅ Sí (disco, ya con LFS) | ✅ Sí |
| Contenedor Docker corriendo | ❌ Se detiene | ❌ Se detiene |
| `netsh portproxy` | ✅ Sí (registro Windows) | ✅ Sí |
| **IP de WSL2** | ✅ Sí (mientras no reinicie WSL) | ❌ **Cambia** |
