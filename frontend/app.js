// ── Configuración ────────────────────────────────────────────
const API_URL = "http://localhost:8000";

// Ids de los 16 campos del formulario (en el mismo orden que FEATURES_BASE del backend)
// El backend agrega internamente las 5 variables derivadas para llegar a 21 features.
const FIELD_IDS = [
  "time_in_hospital", "n_lab_procedures", "n_procedures", "n_medications",
  "n_outpatient", "n_inpatient", "n_emergency", "age_enc",
  "glucose_test_enc", "A1Ctest_enc", "change_enc", "diabetes_med_enc",
  "medical_specialty_enc", "diag_1_enc", "diag_2_enc", "diag_3_enc",
];

// Umbrales del sistema clínico de tres niveles (deben coincidir con el backend)
const T_LOW_MAX = 0.35;   // BAJO:     [0    - 0.35)
const T_MOD_MAX = 0.55;   // MODERADO: [0.35 - 0.55)
                          // ALTO:     [0.55 - 1.00]

// ── Manejo del formulario ─────────────────────────────────────
document.getElementById("patient-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const btn = document.getElementById("btn-predict");
  btn.disabled = true;
  btn.textContent = "⏳ Procesando…";

  // Construir el objeto de datos
  const payload = {};
  for (const id of FIELD_IDS) {
    payload[id] = parseInt(document.getElementById(id).value, 10);
  }

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || `Error ${response.status}`);
    }

    const data = await response.json();
    showResult(data);

  } catch (error) {
    alert(`❌ Error al conectar con el servidor:\n${error.message}\n\nAsegúrate de que el backend esté ejecutándose en ${API_URL}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "🔍 Predecir riesgo de readmisión";
  }
});

// ── Mostrar resultado ─────────────────────────────────────────
function showResult(data) {
  const prob    = data.probability;
  const probPct = Math.round(prob * 100);

  // Indicador de riesgo (sistema de 3 niveles)
  const indicator = document.getElementById("risk-indicator");
  const icon      = document.getElementById("risk-icon");
  const riskText  = document.getElementById("risk-text");
  const probText  = document.getElementById("prob-text");

  indicator.className = "risk-indicator";
  if (prob >= T_MOD_MAX) {
    indicator.classList.add("high");
    icon.textContent     = "🔴";
    riskText.textContent = "Riesgo ALTO de readmisión";
  } else if (prob >= T_LOW_MAX) {
    indicator.classList.add("medium");
    icon.textContent     = "🟠";
    riskText.textContent = "Riesgo MODERADO de readmisión";
  } else {
    indicator.classList.add("low");
    icon.textContent     = "🟢";
    riskText.textContent = "Riesgo BAJO de readmisión";
  }
  probText.textContent = `${probPct} %`;

  // Etiqueta de acción recomendada (viene del backend)
  document.getElementById("risk-label-box").textContent =
    `💡 ${data.recommended_action}`;

  // Barra de probabilidad
  const fill = document.getElementById("prob-bar-fill");
  fill.style.width = "0%";
  setTimeout(() => { fill.style.width = `${probPct}%`; }, 50);

  // Gráfico SHAP — Top 8 variables por valor absoluto
  renderSHAP(data.shap_values, data.features);

  // Timestamp
  document.getElementById("timestamp-text").textContent =
    `Predicción generada el: ${data.timestamp}`;

  // Mostrar el panel de resultados y ocultar el formulario
  document.getElementById("form-section").style.display   = "none";
  document.getElementById("result-section").style.display = "block";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ── Renderizar gráfico SHAP ───────────────────────────────────
function renderSHAP(values, labels) {
  const chartDiv = document.getElementById("shap-chart");
  chartDiv.innerHTML = "";

  // Ordenar por valor absoluto (descendente) y tomar los 8 más importantes
  const indexed = values.map((v, i) => ({ v, label: labels[i] }));
  indexed.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const top8 = indexed.slice(0, 8);

  const maxAbs = Math.max(...top8.map(x => Math.abs(x.v)), 0.001);

  for (const item of top8) {
    const pct      = Math.min((Math.abs(item.v) / maxAbs) * 100, 100);
    const positive = item.v >= 0;
    const sign     = positive ? "+" : "−";

    const row = document.createElement("div");
    row.className = "shap-row";

    row.innerHTML = `
      <span class="shap-label" title="${item.label}">${item.label}</span>
      <div class="shap-bar-wrap">
        <div class="shap-bar ${positive ? "positive" : "negative"}"
             style="width: ${pct}%">
          <span class="shap-val">${sign}${Math.abs(item.v).toFixed(4)}</span>
        </div>
      </div>
    `;
    chartDiv.appendChild(row);
  }
}

// ── Reiniciar formulario ──────────────────────────────────────
function resetForm() {
  document.getElementById("result-section").style.display = "none";
  document.getElementById("form-section").style.display   = "block";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ── Panel de monitoreo ────────────────────────────────────────
async function loadMonitor() {
  const div = document.getElementById("monitor-content");
  div.innerHTML = "<p class='muted'>Cargando estadísticas…</p>";

  try {
    const response = await fetch(`${API_URL}/monitor`);
    if (!response.ok) throw new Error(`Error ${response.status}`);
    const data = await response.json();

    if (data.total_predictions === 0) {
      div.innerHTML = "<p class='muted'>No hay predicciones registradas aún.</p>";
      return;
    }

    // Distribución por nivel (puede no existir si el backend no la calcula aún)
    const dist = data.risk_level_distribution || {
      BAJO:     { count: 0, pct: 0 },
      MODERADO: { count: 0, pct: 0 },
      ALTO:     { count: 0, pct: 0 },
    };

    div.innerHTML = `
      <div class="monitor-grid">
        <div class="monitor-stat">
          <div class="val">${data.total_predictions.toLocaleString("es-CL")}</div>
          <div class="lbl">Predicciones totales</div>
        </div>
        <div class="monitor-stat">
          <div class="val">${data.avg_probability}</div>
          <div class="lbl">Probabilidad media</div>
        </div>
        <div class="monitor-stat">
          <div class="val">${data.min_probability}</div>
          <div class="lbl">Probabilidad mínima</div>
        </div>
        <div class="monitor-stat">
          <div class="val">${data.max_probability}</div>
          <div class="lbl">Probabilidad máxima</div>
        </div>
        <div class="monitor-stat">
          <div class="val">${dist.BAJO.pct} %</div>
          <div class="lbl">Distribución: BAJO</div>
        </div>
        <div class="monitor-stat">
          <div class="val">${dist.MODERADO.pct} %</div>
          <div class="lbl">Distribución: MODERADO</div>
        </div>
        <div class="monitor-stat">
          <div class="val">${dist.ALTO.pct} %</div>
          <div class="lbl">Distribución: ALTO</div>
        </div>
        <div class="monitor-stat">
          <div class="val" style="font-size:1rem">${data.last_prediction_ts}</div>
          <div class="lbl">Última predicción</div>
        </div>
      </div>
    `;
  } catch (error) {
    div.innerHTML = `<p class='muted'>Error al cargar estadísticas: ${error.message}</p>`;
  }
}
