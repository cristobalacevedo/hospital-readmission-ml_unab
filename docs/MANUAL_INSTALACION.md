# Manual de Instalación y Despliegue

**Sistema de Predicción de Readmisión Hospitalaria**
ACIF104 — Aprendizaje de Máquina · UNAB 2026

---

## Índice

1. [Requisitos del sistema](#1-requisitos-del-sistema)
2. [Instalación del entorno de desarrollo](#2-instalación-del-entorno-de-desarrollo)
3. [Generación del modelo (notebook)](#3-generación-del-modelo-notebook)
4. [Despliegue local sin Docker](#4-despliegue-local-sin-docker)
5. [Despliegue con Docker](#5-despliegue-con-docker)
6. [Despliegue como servicio en producción](#6-despliegue-como-servicio-en-producción)
7. [Verificación del despliegue](#7-verificación-del-despliegue)
8. [Operación y monitoreo](#8-operación-y-monitoreo)
9. [Solución de problemas](#9-solución-de-problemas)

---

## 1. Requisitos del Sistema

### 1.1 Requisitos mínimos de hardware

| Componente | Mínimo | Recomendado |
|---|---|---|
| CPU | 2 núcleos | 4 núcleos |
| RAM | 4 GB | 8 GB |
| Disco | 5 GB libres | 10 GB libres |
| Red | Conexión a internet (instalación) | — |

### 1.2 Requisitos de software

| Requisito | Versión mínima | Nota |
|---|---|---|
| Sistema operativo | Linux / macOS / Windows 10+ | WSL2 recomendado para Windows |
| Python | 3.11 | Necesario para todas las dependencias |
| pip | 23.0 | Gestor de paquetes Python |
| Git | 2.30+ | Para clonar el repositorio |
| Navegador web | Chrome 100+ / Firefox 100+ / Edge 100+ | Para acceder al frontend |

### 1.3 Requisitos opcionales (Docker)

| Requisito | Versión mínima |
|---|---|
| Docker Engine | 24.0+ |
| Docker Compose | 2.20+ |

### 1.4 Dependencias Python

Todas las dependencias están listadas en `requirements.txt`:

```
pandas==2.0.3
numpy==1.26.4
scikit-learn==1.4.2
imbalanced-learn==0.11.0
xgboost==2.0.3
lightgbm==4.3.0
torch==2.1.0
shap==0.44.1
fastapi==0.110.0
uvicorn==0.27.1
matplotlib==3.8.3
seaborn==0.13.2
jupyter==1.0.0
joblib==1.3.2
pydantic==2.6.4
```

---

## 2. Instalación del Entorno de Desarrollo

### 2.1 Clonar el repositorio

```bash
git clone https://github.com/cristobalacevedo/hospital-readmission-ml_unab.git
cd hospital-readmission-ml_unab
```

### 2.2 Crear el entorno virtual de Python

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 2.3 Instalar las dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota:** la primera instalación puede tardar entre 5 y 15 minutos debido al tamaño de PyTorch y XGBoost.

### 2.4 Verificar la instalación

```bash
python -c "import torch, sklearn, xgboost, lightgbm, shap, fastapi; print('OK')"
```

Si la salida es `OK`, la instalación es correcta.

---

## 3. Generación del Modelo (Notebook)

Antes de poder ejecutar el backend, es necesario entrenar el modelo y generar los seis artefactos del ensemble.

### 3.1 Obtener el dataset

Descarga `hospital_readmissions.csv` y colócalo en la carpeta `data/`:

- **Fuente UCI:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- **Fuente Kaggle (alternativa):** https://www.kaggle.com/datasets/dubradave/hospital-readmissions

```bash
# El archivo debe quedar en:
data/hospital_readmissions.csv
```

### 3.2 Ejecutar el notebook

```bash
jupyter notebook notebooks/readmision_hospitalaria_colab.ipynb
```

Ejecuta las 16 celdas en orden secuencial (menú "Cell" → "Run All"). El proceso completo tarda entre 5 y 15 minutos según el hardware.

### 3.3 Verificar artefactos generados

Al finalizar el paso 16, deben existir seis archivos en `backend/model/`:

```bash
ls -la backend/model/
```

Salida esperada:

```
rf_final.pkl              (Random Forest entrenado)
xgb_final.pkl             (XGBoost entrenado)
lgb_final.pkl             (LightGBM entrenado)
isotonic_calibrator.pkl   (Calibrador isotónico)
scaler.pkl                (StandardScaler ajustado)
ensemble_config.json      (Pesos y umbrales)
```

---

## 4. Despliegue Local Sin Docker

### 4.1 Opción rápida (script automático)

```bash
./start.sh
```

El script automatiza todos los pasos: crea el entorno, instala dependencias y levanta backend y frontend.

### 4.2 Opción manual paso a paso

#### 4.2.1 Iniciar el backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Si todo va bien verás algo como:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     ✓ Ensemble cargado correctamente
INFO:       • Random Forest   (peso: 0.6518)
INFO:       • XGBoost         (peso: 0.6442)
INFO:       • LightGBM        (peso: 0.6408)
INFO:       • Calibrador isotónico ajustado
INFO:       • Tree SHAP listo sobre el componente Random Forest
```

Verifica en otro terminal:

```bash
curl http://localhost:8000/health
```

#### 4.2.2 Iniciar el frontend

En un terminal separado:

```bash
cd frontend
python3 -m http.server 3000
```

Abre el navegador en: http://localhost:3000

---

## 5. Despliegue con Docker

### 5.1 Construir y levantar el stack completo

```bash
cd docker
docker compose up -d --build
```

Esto crea dos imágenes y levanta dos contenedores:

| Servicio | Imagen | Puerto | Descripción |
|---|---|---|---|
| backend | `readmission-backend:1.0.0` | 8000 | API FastAPI con uvicorn (2 workers) |
| frontend | `readmission-frontend:1.0.0` | 3000 | Nginx sirviendo los archivos estáticos |

### 5.2 Comandos útiles

```bash
# Ver logs en tiempo real
docker compose logs -f

# Ver el estado de cada servicio
docker compose ps

# Reiniciar un servicio
docker compose restart backend

# Detener y eliminar contenedores
docker compose down

# Detener, eliminar y borrar imágenes
docker compose down --rmi all
```

### 5.3 Acceso a los servicios

- **Frontend:** http://localhost:3000
- **Backend:** http://localhost:8000
- **Docs interactiva (Swagger UI):** http://localhost:8000/docs

### 5.4 Persistencia de datos

El `docker-compose.yml` monta dos volúmenes:

```yaml
volumes:
  - ../backend/model:/app/backend/model:ro    # Modelos (solo lectura)
  - ../logs:/app/logs                         # Log de predicciones
```

Esto garantiza que:
- Los modelos persistan entre reinicios del contenedor.
- El log de predicciones se acumule en el host (para análisis posterior).

---

## 6. Despliegue como Servicio en Producción

### 6.1 Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `MODEL_DIR` | `/app/backend/model` | Ubicación de los `.pkl` del ensemble |
| `LOG_PATH` | `/app/logs/predictions.log` | Archivo del log de predicciones |
| `CORS_ORIGINS` | `*` | Orígenes permitidos (separados por coma) |

Ejemplo de uso:

```bash
docker run -d \
  -e CORS_ORIGINS="https://hospital.example.com" \
  -e MODEL_DIR=/app/backend/model \
  -p 8000:8000 \
  readmission-backend:1.0.0
```

### 6.2 Despliegue tras un proxy reverso (Nginx)

Configuración recomendada para servir el sistema bajo HTTPS con dominio propio:

```nginx
# /etc/nginx/sites-available/readmission
server {
    listen 443 ssl http2;
    server_name predict.hospital.example.com;

    ssl_certificate     /etc/letsencrypt/live/predict.hospital.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/predict.hospital.example.com/privkey.pem;

    # Frontend estático
    location / {
        proxy_pass http://localhost:3000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 80;
    server_name predict.hospital.example.com;
    return 301 https://$host$request_uri;
}
```

### 6.3 Despliegue en Kubernetes

Estructura mínima de manifiestos:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: readmission-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: readmission-backend
  template:
    metadata:
      labels:
        app: readmission-backend
    spec:
      containers:
        - name: backend
          image: readmission-backend:1.0.0
          ports:
            - containerPort: 8000
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 20
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
```

### 6.4 Recomendaciones para producción

| Aspecto | Recomendación |
|---|---|
| HTTPS | Obligatorio (Let's Encrypt o certificado institucional) |
| CORS | Restringir a dominios específicos (no `*`) |
| Workers uvicorn | 2× núcleos CPU, mínimo 2 |
| Logging | Centralizar logs (ELK, Grafana Loki) |
| Backup | Respaldar diariamente `backend/model/` y `logs/` |
| Reentrenamiento | Job programado mensual (cron + script) |
| Monitoreo | Prometheus + Grafana sobre `/health` y `/monitor` |

---

## 7. Verificación del Despliegue

### 7.1 Health check del backend

```bash
curl http://localhost:8000/health
```

Salida esperada (HTTP 200):

```json
{
  "status": "ok",
  "model_loaded": true,
  "components": {
    "random_forest":  true,
    "xgboost":        true,
    "lightgbm":       true,
    "calibrator":     true,
    "scaler":         true,
    "shap_explainer": true
  },
  "timestamp": "2026-04-30T15:23:42"
}
```

### 7.2 Información del modelo

```bash
curl http://localhost:8000/model-info
```

### 7.3 Predicción de prueba

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "time_in_hospital": 5,
    "n_lab_procedures": 45,
    "n_procedures": 1,
    "n_medications": 18,
    "n_outpatient": 0,
    "n_inpatient": 2,
    "n_emergency": 0,
    "age_enc": 4,
    "glucose_test_enc": 0,
    "A1Ctest_enc": 2,
    "change_enc": 1,
    "diabetes_med_enc": 1,
    "medical_specialty_enc": 3,
    "diag_1_enc": 2,
    "diag_2_enc": 1,
    "diag_3_enc": 4
  }'
```

### 7.4 Frontend

Abre http://localhost:3000 (o el dominio configurado). Verifica que:

- El formulario carga correctamente con valores por defecto.
- El botón "Predecir riesgo de readmisión" responde sin errores.
- El indicador de riesgo (verde / naranja / rojo) se muestra según la probabilidad.
- El gráfico SHAP aparece con las variables más influyentes.
- El panel "Monitoreo del Modelo" carga estadísticas tras pulsar "Actualizar".

---

## 8. Operación y Monitoreo

### 8.1 Monitoreo de salud (RNF-06)

El endpoint `/monitor` retorna estadísticas agregadas de las predicciones registradas:

```bash
curl http://localhost:8000/monitor | jq
```

Salida ejemplo:

```json
{
  "total_predictions": 1247,
  "avg_probability": 0.4712,
  "min_probability": 0.0823,
  "max_probability": 0.9412,
  "risk_level_distribution": {
    "BAJO":     { "count": 263, "pct": 21.1 },
    "MODERADO": { "count": 576, "pct": 46.2 },
    "ALTO":     { "count": 408, "pct": 32.7 }
  },
  "last_prediction_ts": "2026-04-30T18:42:11",
  "drift_baseline": {
    "BAJO": 21.0, "MODERADO": 46.0, "ALTO": 33.0
  }
}
```

### 8.2 Detección de deriva conceptual

El frontend compara automáticamente la distribución actual con la baseline esperada (BAJO 21 %, MODERADO 46 %, ALTO 33 %) y muestra una alerta si la desviación supera 10 puntos porcentuales en cualquier nivel.

### 8.3 Auditoría de predicciones

Todas las predicciones se registran en `logs/predictions.log` (JSON Lines):

```bash
tail -f logs/predictions.log
```

Cada línea contiene timestamp, las 16 variables de entrada, la probabilidad calibrada, el nivel de riesgo y la predicción binaria.

### 8.4 Reentrenamiento del modelo

Cuando se detecte deriva conceptual o se acumulen suficientes predicciones nuevas:

1. Ejecutar el notebook `notebooks/readmision_hospitalaria_colab.ipynb` con los datos actualizados.
2. Reemplazar los `.pkl` en `backend/model/`.
3. Reiniciar el backend:

```bash
docker compose restart backend
```

---

## 9. Solución de Problemas

### 9.1 El backend no arranca: "FileNotFoundError: rf_final.pkl"

**Causa:** los artefactos del modelo no están generados.

**Solución:** ejecutar el notebook (paso 16) o copiar los `.pkl` desde otro entorno.

### 9.2 El frontend no se conecta al backend

**Causa típica:** CORS bloqueado o backend no accesible.

**Solución:**

```bash
# Verificar que el backend responde
curl http://localhost:8000/health

# Ajustar la URL del API si es necesario (en navegador):
# http://localhost:3000/?api=http://otro-host:8000
```

### 9.3 Error de memoria al cargar XGBoost / LightGBM

**Causa:** RAM insuficiente.

**Solución:** mínimo 4 GB libres. Cierra otras aplicaciones o aumenta la memoria del contenedor:

```yaml
# docker-compose.yml
services:
  backend:
    mem_limit: 2g
```

### 9.4 Las predicciones son siempre las mismas

**Causa:** posiblemente caché del navegador o modelo desactualizado.

**Solución:**

```bash
# Forzar recarga del navegador (Ctrl+Shift+R)
# Verificar la versión del modelo
curl http://localhost:8000/model-info
```

### 9.5 Error "ModuleNotFoundError" al ejecutar el notebook

**Causa:** entorno virtual no activado o dependencias faltantes.

**Solución:**

```bash
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

### 9.6 Logs llenando el disco

**Solución:** rotar el log periódicamente con `logrotate`:

```
# /etc/logrotate.d/readmission
/path/to/hospital-readmission-ml/logs/predictions.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
    copytruncate
}
```

---

## Anexo — Comandos de Referencia Rápida

```bash
# Ciclo de vida completo
./start.sh                        # Arranque local
docker compose up -d              # Arranque con Docker
docker compose logs -f            # Ver logs
docker compose ps                 # Estado
docker compose down               # Detener

# Diagnóstico
curl http://localhost:8000/health
curl http://localhost:8000/model-info
curl http://localhost:8000/monitor

# Tests del sistema
pytest tests/                     # (cuando existan tests)

# Desarrollo
jupyter notebook                  # Notebook interactivo
uvicorn main:app --reload         # Backend en modo desarrollo
```

---

**Equipo 3 — ACIF104 Aprendizaje de Máquina · UNAB 2026**
Repositorio: https://github.com/cristobalacevedo/hospital-readmission-ml_unab
