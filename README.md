# 🏥 Predicción de Readmisión Hospitalaria en Pacientes Diabéticos

## ACIF104 — Aprendizaje de Máquina NRC 2182 | UNAB 

Este proyecto implementa un pipeline avanzado de Machine Learning y Deep Learning (PyTorch) para predecir la readmisión hospitalaria. El modelo de Red Neuronal está optimizado para procesamiento en paralelo mediante **NVIDIA CUDA**.

Proyecto de Machine Learning para predecir la readmisión hospitalaria en pacientes diabéticos.
Dataset: UCI Diabetes 130-US Hospitals (1999–2008).

## ⚠️ Requisito de Hardware (IMPORTANTE)
Este proyecto utiliza **CUDA** para acelerar el entrenamiento de la red neuronal y el cálculo de valores SHAP complejos. 
* **Entorno Obligatorio:** NVIDIA GPU con soporte CUDA.
* **Plataforma recomendada:** Google Colab (con aceleración por hardware T4/L4 activada).

## Estructura del Repositorio

```
hospital-readmission-ml/
  README.md
  requirements.txt
  .gitignore
  data/
    hospital_readmissions.csv
    README.md             (sobre el dataset)
  notebooks/
    01_EDA.ipynb
    02_Models.ipynb
    03_DL.ipynb
    04_SHAP.ipynb
    readmision_hospitalaria_colab.ipynb
  backend/                Directorio provisorio para siguiente etapa de montaje en web
    main.py
    model/                Generado al ejecutar notebook
  frontend/               Directorio provisorio para siguiente etapa de montaje en web
    index.html
    style.css
    app.js
  logs/                   Generado automáticamente en runtime
```

## Instalación Rápida

```bash
git clone https://github.com/grupo-acif104/hospital-readmission-ml_unab.git
cd hospital-readmission-ml_unab
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Obtener el Dataset

Descarga `hospital_readmissions.csv` y colócalo en la carpeta `data/`.
Fuente: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

## Ejecutar los Notebooks

```bash
jupyter notebook
```

Ejecutar: 
1. `notebooks/readmision_hospitalaria_colab.ipynb` (dentro de Colab) ← Genera `/modelos/rf_model.pkl` y `scaler.pkl`

o en su defecto, por partes:

1. `notebooks/01_EDA.ipynb`
2. `notebooks/02_Models.ipynb`  ← Genera `backend/model/rf_model.pkl` y `scaler.pkl`
3. `notebooks/03_DL.ipynb`
4. `notebooks/04_SHAP.ipynb`

## Ejecutar el Backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API disponible en: http://localhost:8000  
Documentación (Swagger): http://localhost:8000/docs

## Ejecutar el Frontend

```bash
cd frontend
python -m http.server 3000
```

Abrir en el navegador: http://localhost:3000

## Tecnologías

- NVIDIA CUDA GPU T4 - Google Colab
- Python 3.11, PyTorch 2.1, scikit-learn 1.4, imbalanced-learn 0.11
- SHAP 0.44, FastAPI 0.110, uvicorn 0.27
- HTML5, CSS3, JavaScript (vanilla)

## Equipo

Proyecto grupal — ACIF104 Aprendizaje de Máquina NRC 2182 — UNAB 2025
