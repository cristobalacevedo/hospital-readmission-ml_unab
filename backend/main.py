"""
Backend API REST — Predicción de Readmisión Hospitalaria
ACIF104 — Aprendizaje Automático | UNAB 2025

Ejecutar:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Documentación interactiva:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import shap
import json
import datetime
import os

# ── Configuración de la aplicación ────────────────────────────
app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="API REST para predecir la readmisión hospitalaria en pacientes diabéticos.",
    version="1.0.0",
)

# Habilitar CORS para el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Carga del modelo y el escalador ───────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "model", "rf_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")
LOG_PATH    = os.path.join(os.path.dirname(__file__), "..", "logs", "predictions.log")

try:
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    explainer = shap.TreeExplainer(model)
    print("✓ Modelo y escalador cargados correctamente.")
except FileNotFoundError as e:
    print(f"✗ Error al cargar el modelo: {e}")
    print("  Ejecuta primero el notebook 02_ML_Models.ipynb para generarlo.")
    model = None
    scaler = None
    explainer = None

# Lista de variables (debe coincidir con el orden del entrenamiento)
FEATURES = [
    "time_in_hospital", "n_lab_procedures", "n_procedures", "n_medications",
    "n_outpatient", "n_inpatient", "n_emergency", "age_enc",
    "glucose_test_enc", "A1Ctest_enc", "change_enc", "diabetes_med_enc",
    "medical_specialty_enc", "diag_1_enc", "diag_2_enc", "diag_3_enc",
]

FEATURE_LABELS = [
    "Días en hospital", "Procedimientos de lab.", "Procedimientos clínicos",
    "Medicamentos", "Visitas ambulatorias", "Ingresos previos", "Visitas a urgencias",
    "Edad (codificada)", "Test de glucosa", "Test HbA1c", "Cambio de medicamento",
    "Medicamento para diabetes", "Especialidad médica", "Diagnóstico 1",
    "Diagnóstico 2", "Diagnóstico 3",
]


# ── Esquema de entrada ─────────────────────────────────────────
class PatientData(BaseModel):
    time_in_hospital:      int = Field(..., ge=1, le=14,  description="Días de hospitalización (1–14)")
    n_lab_procedures:      int = Field(..., ge=0, le=150, description="N.º de procedimientos de laboratorio")
    n_procedures:          int = Field(..., ge=0, le=10,  description="N.º de procedimientos clínicos")
    n_medications:         int = Field(..., ge=0, le=100, description="N.º de medicamentos prescritos")
    n_outpatient:          int = Field(..., ge=0, le=50,  description="Visitas ambulatorias previas")
    n_inpatient:           int = Field(..., ge=0, le=20,  description="Hospitalizaciones previas (predictor clave)")
    n_emergency:           int = Field(..., ge=0, le=70,  description="Visitas a urgencias previas")
    age_enc:               int = Field(..., ge=0, le=5,   description="Grupo etario codificado (0=[40-50) … 5=[90-100))")
    glucose_test_enc:      int = Field(..., ge=0, le=2,   description="Resultado test glucosa (0=no, 1=normal, 2=high)")
    A1Ctest_enc:           int = Field(..., ge=0, le=2,   description="Resultado HbA1c (0=no, 1=normal, 2=high)")
    change_enc:            int = Field(..., ge=0, le=1,   description="Cambio de medicamento (0=no, 1=yes)")
    diabetes_med_enc:      int = Field(..., ge=0, le=1,   description="Medicamento para diabetes (0=no, 1=yes)")
    medical_specialty_enc: int = Field(..., ge=0, le=6,   description="Especialidad médica codificada (0–6)")
    diag_1_enc:            int = Field(..., ge=0, le=7,   description="Diagnóstico primario codificado (0–7)")
    diag_2_enc:            int = Field(..., ge=0, le=7,   description="Diagnóstico secundario codificado (0–7)")
    diag_3_enc:            int = Field(..., ge=0, le=7,   description="Diagnóstico terciario codificado (0–7)")

    class Config:
        json_schema_extra = {
            "example": {
                "time_in_hospital": 5, "n_lab_procedures": 45, "n_procedures": 1,
                "n_medications": 18, "n_outpatient": 0, "n_inpatient": 2,
                "n_emergency": 0, "age_enc": 4, "glucose_test_enc": 0,
                "A1Ctest_enc": 2, "change_enc": 1, "diabetes_med_enc": 1,
                "medical_specialty_enc": 3, "diag_1_enc": 2, "diag_2_enc": 1, "diag_3_enc": 4
            }
        }


# ── Esquema de respuesta ───────────────────────────────────────
class PredictionResponse(BaseModel):
    readmitted:   int
    probability:  float
    risk_label:   str
    shap_values:  list
    features:     list
    timestamp:    str


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/", summary="Estado del servicio")
def root():
    return {
        "service": "Hospital Readmission Prediction API",
        "version": "1.0.0",
        "status":  "online" if model is not None else "model_not_loaded",
        "docs":    "/docs",
    }


@app.get("/health", summary="Verificación de salud del servicio")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, summary="Predecir readmisión hospitalaria")
def predict(data: PatientData):
    """
    Recibe los 16 atributos clínicos de un paciente y retorna:
    - **readmitted**: 1 si se predice readmisión, 0 en caso contrario.
    - **probability**: probabilidad de readmisión (0,0 – 1,0).
    - **risk_label**: etiqueta de riesgo (Alto / Moderado / Bajo).
    - **shap_values**: contribución de cada variable a la predicción (Tree SHAP).
    - **features**: nombres de las variables en el mismo orden que shap_values.
    - **timestamp**: fecha y hora de la predicción.
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está disponible. Ejecuta primero el notebook 02_ML_Models.ipynb."
        )

    # Preparar el vector de entrada
    values = [getattr(data, f) for f in FEATURES]
    X_raw  = np.array(values, dtype=np.float32).reshape(1, -1)
    X_sc   = scaler.transform(X_raw)

    # Predicción
    prob = float(model.predict_proba(X_sc)[0][1])

    # Etiqueta de riesgo
    if prob >= 0.60:
        risk_label = "Alto — Intervención preventiva recomendada"
    elif prob >= 0.40:
        risk_label = "Moderado — Seguimiento reforzado recomendado"
    else:
        risk_label = "Bajo — Seguimiento estándar"

    # Valores SHAP
    sv_raw = explainer.shap_values(X_sc)
    if isinstance(sv_raw, list):
        sv = sv_raw[1][0].tolist()   # Clase positiva
    else:
        sv = sv_raw[0].tolist()

    # Registro del log
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    log_entry = {
        "ts":          timestamp,
        "probability": round(prob, 4),
        "prediction":  int(prob > 0.5),
        "risk_label":  risk_label,
        **{f: getattr(data, f) for f in FEATURES},
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return PredictionResponse(
        readmitted  = int(prob > 0.5),
        probability = round(prob, 4),
        risk_label  = risk_label,
        shap_values = [round(v, 6) for v in sv],
        features    = FEATURE_LABELS,
        timestamp   = timestamp,
    )


@app.get("/monitor", summary="Estadísticas de las predicciones registradas")
def monitor():
    """Retorna estadísticas resumidas del log de predicciones para monitoreo del desempeño."""
    if not os.path.exists(LOG_PATH):
        return {"total_predictions": 0, "message": "No hay predicciones registradas aún."}

    entries = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not entries:
        return {"total_predictions": 0}

    probs = [e["probability"] for e in entries]
    preds = [e["prediction"] for e in entries]

    return {
        "total_predictions":     len(entries),
        "readmission_rate_pct":  round(sum(preds) / len(preds) * 100, 2),
        "avg_probability":       round(sum(probs) / len(probs), 4),
        "min_probability":       round(min(probs), 4),
        "max_probability":       round(max(probs), 4),
        "last_prediction_ts":    entries[-1].get("ts", "N/A"),
    }
