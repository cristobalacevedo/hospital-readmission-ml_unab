# Hospital Readmission Prediction — ACIF104

Sistema de predicción de readmisión hospitalaria en pacientes diabéticos basado en aprendizaje automático, con explicabilidad clínica mediante SHAP y arquitectura de despliegue completa (backend FastAPI + frontend web).

**Asignatura:** ACIF104 — Aprendizaje de Máquina · UNAB 2026

## Modelo del Sistema

El sistema utiliza un **ensemble ponderado calibrado** que combina tres clasificadores complementarios basados en árboles:

- **Random Forest** — explicabilidad exacta mediante Tree SHAP
- **XGBoost** — captura interacciones no lineales con regularización L1/L2
- **LightGBM** — diversidad metodológica con crecimiento por hojas

Las probabilidades del ensemble se procesan mediante **calibración isotónica** ajustada en validación, garantizando interpretación clínica directa de las probabilidades emitidas.

### Sistema clínico de tres niveles de riesgo

| Nivel | Rango de probabilidad | Protocolo clínico |
|-------|-----------------------|-------------------|
| 🟢 BAJO | `[0 — 0,35)` | Seguimiento ambulatorio estándar; educación en autocuidado |
| 🟠 MODERADO | `[0,35 — 0,55)` | Seguimiento telefónico a 7 días; revisión farmacológica |
| 🔴 ALTO | `[0,55 — 1,00]` | Intervención preventiva inmediata; visita domiciliaria a 48 h |

**Umbral clínico binario:** `0,42` (Recall ≥ 0,85 sobre la clase positiva).

## Estructura del Repositorio

```
hospital-readmission-ml/
  README.md
  requirements.txt
  .gitignore
  data/                  ← Coloca aquí hospital_readmissions.csv
  notebooks/
    readmision_hospitalaria_colab.ipynb       ← Pipeline completo
  backend/
    main.py              ← FastAPI con ensemble calibrado
    model/               ← Generado por el notebook (paso 16)
      rf_final.pkl
      xgb_final.pkl
      lgb_final.pkl
      isotonic_calibrator.pkl
      scaler.pkl
      ensemble_config.json
  frontend/
    index.html
    style.css
    app.js
  logs/                  ← Generado automáticamente en runtime
```

## Instalación

```bash
git clone https://github.com/cristobalacevedo/hospital-readmission-ml_unab.git
cd hospital-readmission-ml_unab
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Obtener el Dataset

Descarga `hospital_readmissions.csv` y colócalo en la carpeta `data/`.
Fuente: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

## Ejecutar el Notebook

```bash
jupyter notebook notebooks/readmision_hospitalaria_colab.ipynb
```

El notebook ejecuta el pipeline completo en 16 pasos secuenciales:

1. Instalación de dependencias
2. Carga del dataset
3. Importaciones y configuración global
4. Preprocesamiento de datos
5. Análisis exploratorio (EDA)
6. Partición estratificada y balanceo SMOTE
7. Modelos base (LR, RF, SVM)
8. Red Neuronal MLP con PyTorch
9. Comparación de modelos base
10. Ingeniería de características avanzada (5 variables derivadas)
11. Modelos del ensemble (RF optimizado, XGBoost, LightGBM)
12. Ensemble ponderado con calibración isotónica
13. Validación cruzada estratificada 5-fold
14. Explicabilidad con Tree SHAP
15. Sistema de predicción interactivo con 3 niveles de riesgo
16. Guardado de modelos para despliegue

Al finalizar, los artefactos del modelo quedarán en `backend/model/`.

## Ejecutar el Backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API disponible en: http://localhost:8000
Documentación interactiva (Swagger): http://localhost:8000/docs

### Endpoints expuestos

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Estado general del servicio |
| GET | `/health` | Estado de carga de cada componente del ensemble |
| GET | `/model-info` | Configuración del ensemble (pesos y umbrales) |
| POST | `/predict` | Predicción con sistema de 3 niveles y SHAP |
| GET | `/monitor` | Estadísticas agregadas del log de predicciones |

## Ejecutar el Frontend

```bash
cd frontend
python -m http.server 3000
```

Abrir en el navegador: http://localhost:3000

## Tecnologías

- Python 3.11, PyTorch 2.1, scikit-learn 1.4, imbalanced-learn 0.11
- XGBoost 2.0, LightGBM 4.3
- SHAP 0.44, FastAPI 0.110, uvicorn 0.27
- HTML5, CSS3, JavaScript (vanilla)

## Equipo

Proyecto grupal — ACIF104 Aprendizaje de Máquina — UNAB 2026
