"""
Backend API REST — Predicción de Readmisión Hospitalaria
ACIF104 — Aprendizaje de Máquina | UNAB 2026

Modelo: Ensemble ponderado de Random Forest + XGBoost + LightGBM,
        con calibración isotónica y sistema clínico de tres niveles de riesgo.

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
import pandas as pd
import shap
import json
import datetime
import os

# ── Configuración de la aplicación ────────────────────────────
app = FastAPI(
    title="Hospital Readmission Prediction API",
    description=(
        "API REST para predecir la readmisión hospitalaria en pacientes "
        "diabéticos usando un ensemble ponderado calibrado (Random Forest + "
        "XGBoost + LightGBM + calibración isotónica) con sistema clínico "
        "de tres niveles de riesgo."
    ),
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

# ── Rutas de los artefactos del modelo ────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
RF_PATH    = os.path.join(MODEL_DIR, "rf_final.pkl")
XGB_PATH   = os.path.join(MODEL_DIR, "xgb_final.pkl")
LGB_PATH   = os.path.join(MODEL_DIR, "lgb_final.pkl")
ISO_PATH   = os.path.join(MODEL_DIR, "isotonic_calibrator.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "ensemble_config.json")
LOG_PATH   = os.path.join(os.path.dirname(__file__), "..", "logs", "predictions.log")

# ── Carga de los componentes del ensemble ─────────────────────
try:
    rf_final  = joblib.load(RF_PATH)
    xgb_final = joblib.load(XGB_PATH)
    lgb_final = joblib.load(LGB_PATH)
    isotonic  = joblib.load(ISO_PATH)
    scaler    = joblib.load(SCALER_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)

    # Tree SHAP sobre el componente Random Forest del ensemble
    explainer = shap.TreeExplainer(rf_final)

    print("✓ Ensemble cargado correctamente:")
    print(f"  - Random Forest   (peso: {CONFIG['weights']['rf']:.4f})")
    print(f"  - XGBoost         (peso: {CONFIG['weights']['xgb']:.4f})")
    print(f"  - LightGBM        (peso: {CONFIG['weights']['lgb']:.4f})")
    print(f"  - Calibrador isotónico ajustado")
    print(f"  - Tree SHAP sobre RF para explicabilidad")
    MODEL_LOADED = True
except FileNotFoundError as e:
    print(f"✗ Error al cargar los artefactos del ensemble: {e}")
    print("  Ejecuta primero el notebook readmision_hospitalaria_colab.ipynb")
    print("  (paso 16) para generar los archivos en el directorio model/.")
    rf_final = xgb_final = lgb_final = None
    isotonic = scaler = explainer = None
    CONFIG = {
        "weights": {"rf": 0.0, "xgb": 0.0, "lgb": 0.0},
        "features_base": [],
        "features_derived": [],
        "features_ext": [],
        "thresholds": {"low_max": 0.35, "mod_max": 0.55, "clinical": 0.42},
    }
    MODEL_LOADED = False

# ── Variables originales (16 features) ────────────────────────
FEATURES_BASE = [
    "time_in_hospital", "n_lab_procedures", "n_procedures", "n_medications",
    "n_outpatient", "n_inpatient", "n_emergency", "age_enc",
    "glucose_test_enc", "A1Ctest_enc", "change_enc", "diabetes_med_enc",
    "medical_specialty_enc", "diag_1_enc", "diag_2_enc", "diag_3_enc",
]

# ── Variables derivadas (5 features) ──────────────────────────
FEATURES_DERIVED = [
    "complexity", "utilizacion_prev", "proc_per_day",
    "med_intensity", "risk_score_base",
]

# ── Conjunto extendido (21 features) ──────────────────────────
FEATURES_EXT = FEATURES_BASE + FEATURES_DERIVED

FEATURE_LABELS_EXT = [
    "Días en hospital", "Procedimientos de lab.", "Procedimientos clínicos",
    "Medicamentos", "Visitas ambulatorias", "Ingresos previos", "Visitas a urgencias",
    "Edad (codificada)", "Test de glucosa", "Test HbA1c", "Cambio de medicamento",
    "Medicamento para diabetes", "Especialidad médica", "Diagnóstico 1",
    "Diagnóstico 2", "Diagnóstico 3",
    "Complejidad", "Utilización previa", "Proc. por día",
    "Intensidad de medicación", "Score de riesgo",
]

# Pesos del ensemble y umbrales clínicos
W_RF  = CONFIG["weights"]["rf"]
W_XGB = CONFIG["weights"]["xgb"]
W_LGB = CONFIG["weights"]["lgb"]
WSUM  = W_RF + W_XGB + W_LGB if MODEL_LOADED else 1.0

T_LOW_MAX  = CONFIG["thresholds"]["low_max"]    # 0.35
T_MOD_MAX  = CONFIG["thresholds"]["mod_max"]    # 0.55
T_CLINICAL = CONFIG["thresholds"]["clinical"]   # 0.42 (Recall >= 0.85)


# =============================================================
# Feature engineering (idéntico al del notebook)
# =============================================================
def build_features_extended(data: dict) -> np.ndarray:
    """
    Agrega las 5 variables derivadas a los 16 features originales,
    aplica StandardScaler y retorna el vector listo para predicción.

    Las variables derivadas son:
      - complexity       = n_inpatient * n_medications
      - utilizacion_prev = n_inpatient + n_outpatient + n_emergency
      - proc_per_day     = (n_procedures + n_lab_procedures) / time_in_hospital
      - med_intensity    = n_medications / time_in_hospital
      - risk_score_base  = 3*n_inpatient + 2*n_emergency + 2*A1C + change
    """
    df = pd.DataFrame([data])

    df["complexity"]       = df["n_inpatient"] * df["n_medications"]
    df["utilizacion_prev"] = (df["n_inpatient"] + df["n_outpatient"]
                              + df["n_emergency"])
    df["proc_per_day"]     = ((df["n_procedures"] + df["n_lab_procedures"])
                              / (df["time_in_hospital"] + 1e-6))
    df["med_intensity"]    = (df["n_medications"]
                              / (df["time_in_hospital"] + 1e-6))
    df["risk_score_base"]  = (df["n_inpatient"] * 3
                              + df["n_emergency"] * 2
                              + df["A1Ctest_enc"] * 2
                              + df["change_enc"])

    X_ext = df[FEATURES_EXT].values.astype(np.float32)
    return scaler.transform(X_ext)


# =============================================================
# Predicción del ensemble ponderado con calibración isotónica
# =============================================================
def predict_ensemble(X_scaled: np.ndarray) -> float:
    """Retorna la probabilidad calibrada del ensemble para un paciente."""
    p_rf  = rf_final.predict_proba(X_scaled)[:, 1]
    p_xgb = xgb_final.predict_proba(X_scaled)[:, 1]
    p_lgb = lgb_final.predict_proba(X_scaled)[:, 1]

    p_raw = (p_rf * W_RF + p_xgb * W_XGB + p_lgb * W_LGB) / WSUM
    p_cal = isotonic.predict(p_raw)
    return float(p_cal[0])


def classify_risk_level(probability: float) -> dict:
    """
    Clasifica el paciente en uno de los tres niveles clínicos de riesgo.
    Retorna nivel, etiqueta completa y protocolo recomendado.
    """
    if probability >= T_MOD_MAX:       # >= 0.55
        return {
            "level": "ALTO",
            "emoji": "🔴",
            "label": "Riesgo ALTO de readmisión",
            "action": "Intervención preventiva inmediata; visita domiciliaria a 48 h",
            "css_class": "high",
        }
    elif probability >= T_LOW_MAX:      # 0.35 - 0.55
        return {
            "level": "MODERADO",
            "emoji": "🟠",
            "label": "Riesgo MODERADO de readmisión",
            "action": "Seguimiento telefónico a 7 días; revisión farmacológica con atención primaria",
            "css_class": "moderate",
        }
    else:                                # < 0.35
        return {
            "level": "BAJO",
            "emoji": "🟢",
            "label": "Riesgo BAJO de readmisión",
            "action": "Seguimiento ambulatorio estándar; educación en autocuidado al alta",
            "css_class": "low",
        }


# =============================================================
# Esquemas Pydantic
# =============================================================
class PatientData(BaseModel):
    time_in_hospital:      int = Field(..., ge=1, le=14,  description="Días de hospitalización (1–14)")
    n_lab_procedures:      int = Field(..., ge=0, le=150, description="N.º de procedimientos de laboratorio")
    n_procedures:          int = Field(..., ge=0, le=10,  description="N.º de procedimientos clínicos")
    n_medications:         int = Field(..., ge=0, le=100, description="N.º de medicamentos prescritos")
    n_outpatient:          int = Field(..., ge=0, le=50,  description="Visitas ambulatorias previas")
    n_inpatient:           int = Field(..., ge=0, le=20,  description="Hospitalizaciones previas (predictor clave)")
    n_emergency:           int = Field(..., ge=0, le=70,  description="Visitas a urgencias previas")
    age_enc:               int = Field(..., ge=0, le=5,   description="Grupo etario codificado (0=[40-50) ... 5=[90-100))")
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
                "medical_specialty_enc": 3, "diag_1_enc": 2, "diag_2_enc": 1, "diag_3_enc": 4,
            }
        }


class PredictionResponse(BaseModel):
    probability:       float
    risk_level:        str
    risk_emoji:        str
    risk_label:        str
    recommended_action: str
    readmitted_binary:  int
    shap_values:       list
    features:          list
    model_info:        dict
    timestamp:         str


# =============================================================
# Endpoints
# =============================================================
@app.get("/", summary="Estado del servicio")
def root():
    return {
        "service": "Hospital Readmission Prediction API",
        "version": "1.0.0",
        "model":   "Ensemble Ponderado Calibrado (RF + XGBoost + LightGBM)",
        "status":  "online" if MODEL_LOADED else "model_not_loaded",
        "docs":    "/docs",
    }


@app.get("/health", summary="Verificación de salud del servicio")
def health():
    return {
        "status":       "ok",
        "model_loaded": MODEL_LOADED,
        "components":   {
            "random_forest":  rf_final is not None,
            "xgboost":        xgb_final is not None,
            "lightgbm":       lgb_final is not None,
            "calibrator":     isotonic is not None,
            "scaler":         scaler is not None,
        },
    }


@app.get("/model-info", summary="Información del modelo desplegado")
def model_info():
    """Expone la configuración del ensemble y los umbrales clínicos."""
    return {
        "name":    "Ensemble Ponderado Calibrado",
        "version": "1.0.0",
        "components": [
            {"name": "Random Forest", "weight": W_RF / WSUM if WSUM else 0.0},
            {"name": "XGBoost",       "weight": W_XGB / WSUM if WSUM else 0.0},
            {"name": "LightGBM",      "weight": W_LGB / WSUM if WSUM else 0.0},
        ],
        "calibration": "Isotonic Regression (sobre validación)",
        "explainability": "Tree SHAP sobre el componente Random Forest",
        "thresholds": {
            "low_max":         T_LOW_MAX,
            "moderate_max":    T_MOD_MAX,
            "clinical_binary": T_CLINICAL,
        },
        "features_count": {
            "base":     len(FEATURES_BASE),
            "derived":  len(FEATURES_DERIVED),
            "total":    len(FEATURES_EXT),
        },
    }


@app.post("/predict", response_model=PredictionResponse,
          summary="Predecir readmisión hospitalaria con sistema de 3 niveles")
def predict(data: PatientData):
    """
    Recibe los 16 atributos clínicos de un paciente y retorna:

    - **probability**: probabilidad calibrada de readmisión (0,0 – 1,0).
    - **risk_level**: nivel clínico de riesgo (BAJO / MODERADO / ALTO).
    - **risk_emoji**: indicador visual (🟢 / 🟠 / 🔴).
    - **risk_label**: etiqueta descriptiva del nivel.
    - **recommended_action**: protocolo clínico recomendado.
    - **readmitted_binary**: clasificación binaria con umbral clínico 0,42 (Recall >= 0,85).
    - **shap_values**: contribución de cada variable a la predicción (Tree SHAP).
    - **features**: nombres de las 21 variables en el mismo orden que shap_values.
    - **model_info**: metadatos del ensemble que produjo la predicción.
    - **timestamp**: fecha y hora de la predicción.
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo no está disponible. Ejecuta primero el notebook "
                "readmision_hospitalaria_colab.ipynb (paso 16) para generar "
                "los archivos del ensemble en el directorio model/."
            ),
        )

    # Construir vector extendido (21 features) y aplicar escalado
    X_scaled = build_features_extended(data.dict())

    # Predicción del ensemble calibrado
    probability = predict_ensemble(X_scaled)

    # Clasificación en 3 niveles
    risk = classify_risk_level(probability)

    # Clasificación binaria clínica (umbral 0.42 -> Recall >= 0.85)
    readmitted_binary = int(probability >= T_CLINICAL)

    # Valores SHAP sobre el componente Random Forest del ensemble
    sv_raw = explainer.shap_values(X_scaled)
    if isinstance(sv_raw, list):
        sv = sv_raw[1][0].tolist()
    elif sv_raw.ndim == 3:
        sv = sv_raw[0, :, 1].tolist()
    else:
        sv = sv_raw[0].tolist()

    # Registro del log
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    log_entry = {
        "ts":           timestamp,
        "probability":  round(probability, 4),
        "risk_level":   risk["level"],
        "binary_pred":  readmitted_binary,
        **{f: getattr(data, f) for f in FEATURES_BASE},
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return PredictionResponse(
        probability        = round(probability, 4),
        risk_level         = risk["level"],
        risk_emoji         = risk["emoji"],
        risk_label         = risk["label"],
        recommended_action = risk["action"],
        readmitted_binary  = readmitted_binary,
        shap_values        = [round(v, 6) for v in sv],
        features           = FEATURE_LABELS_EXT,
        model_info         = {
            "ensemble":           "RF + XGBoost + LightGBM",
            "calibration":        "Isotonic",
            "clinical_threshold": T_CLINICAL,
            "risk_levels":    {
                "BAJO":     f"[0 — {T_LOW_MAX})",
                "MODERADO": f"[{T_LOW_MAX} — {T_MOD_MAX})",
                "ALTO":     f"[{T_MOD_MAX} — 1]",
            },
        },
        timestamp          = timestamp,
    )


@app.get("/monitor", summary="Estadísticas agregadas del log de predicciones")
def monitor():
    """
    Retorna estadísticas resumidas del log de predicciones, útiles para
    el monitoreo del desempeño del sistema y detección de deriva conceptual.
    """
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
    levels = [e.get("risk_level", "N/A") for e in entries]

    # Conteo por nivel de riesgo
    counts = {"BAJO": 0, "MODERADO": 0, "ALTO": 0}
    for lv in levels:
        if lv in counts:
            counts[lv] += 1
    total = len(entries)

    return {
        "total_predictions":     total,
        "avg_probability":       round(sum(probs) / total, 4),
        "min_probability":       round(min(probs), 4),
        "max_probability":       round(max(probs), 4),
        "risk_level_distribution": {
            "BAJO":     {"count": counts["BAJO"],     "pct": round(counts["BAJO"]     / total * 100, 1)},
            "MODERADO": {"count": counts["MODERADO"], "pct": round(counts["MODERADO"] / total * 100, 1)},
            "ALTO":     {"count": counts["ALTO"],     "pct": round(counts["ALTO"]     / total * 100, 1)},
        },
        "last_prediction_ts":    entries[-1].get("ts", "N/A"),
    }
