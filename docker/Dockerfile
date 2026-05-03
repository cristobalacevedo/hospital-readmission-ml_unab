# ═══════════════════════════════════════════════════════════════════
# Dockerfile — Backend FastAPI
# Predicción de Readmisión Hospitalaria
# ACIF104 — Aprendizaje de Máquina · UNAB 2026
# ═══════════════════════════════════════════════════════════════════
FROM python:3.11-slim

# Variables de entorno para Python en contenedor
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencias del sistema (necesarias para xgboost / lightgbm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no privilegiado para mayor seguridad
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Instalar dependencias Python (cacheable layer)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copiar el código del backend y los modelos
COPY backend/ /app/backend/
COPY logs/    /app/logs/

# Permisos para el usuario no-root
RUN chown -R appuser:appuser /app
USER appuser

# Configuración de salida
EXPOSE 8000

# Healthcheck que consulta /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Punto de entrada
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
