/* ═══════════════════════════════════════════════════════════════
   Sistema de Predicción de Readmisión Hospitalaria
   Lógica del cliente — invoca la API del backend
   ACIF104 — Aprendizaje de Máquina · UNAB 2026
═══════════════════════════════════════════════════════════════ */

'use strict';

// ── Configuración ────────────────────────────────────────────
// Permite override desde la URL: ?api=https://otro-host:8000
const urlParams = new URLSearchParams(window.location.search);
const API_URL = urlParams.get('api') || 'http://localhost:8000';

// Identificadores de los 16 campos del formulario,
// en el orden esperado por el backend (FEATURES_BASE).
const FIELD_IDS = [
  'time_in_hospital', 'n_lab_procedures', 'n_procedures', 'n_medications',
  'n_outpatient', 'n_inpatient', 'n_emergency', 'age_enc',
  'glucose_test_enc', 'A1Ctest_enc', 'change_enc', 'diabetes_med_enc',
  'medical_specialty_enc', 'diag_1_enc', 'diag_2_enc', 'diag_3_enc',
];

// Umbrales del sistema clínico de tres niveles
// (deben coincidir con los del backend / ensemble_config.json)
const T_LOW_MAX = 0.35;   // BAJO     [0    – 0,35)
const T_MOD_MAX = 0.55;   // MODERADO [0,35 – 0,55)
                          // ALTO     [0,55 – 1,00]

// ═════════════════════════════════════════════════════════════
// 1. INICIALIZACIÓN — health check al cargar la página
// ═════════════════════════════════════════════════════════════
window.addEventListener('DOMContentLoaded', async () => {
  try {
    const resp = await fetch(`${API_URL}/health`, { method: 'GET' });
    if (!resp.ok) {
      showApiWarning(`El backend respondió con código ${resp.status}.`);
      return;
    }
    const data = await resp.json();
    if (!data.model_loaded) {
      showApiWarning('El modelo no está cargado. Ejecuta el notebook (paso 16) para generar los artefactos del ensemble.');
    }
  } catch (err) {
    showApiWarning(`No se pudo conectar al backend (${API_URL}). Asegúrate de que el servicio esté ejecutándose.`);
  }
});

function showApiWarning(message) {
  const formSection = document.getElementById('form-section');
  const banner = document.createElement('div');
  banner.style.cssText = `
    background: #FEF3C7; color: #92400E; border-left: 4px solid var(--warning);
    padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;
    font-size: 0.88rem;
  `;
  banner.innerHTML = `⚠️ <b>Aviso:</b> ${message}`;
  formSection.insertBefore(banner, formSection.firstChild.nextSibling);
}

// ═════════════════════════════════════════════════════════════
// 2. FORMULARIO — envío a /predict
// ═════════════════════════════════════════════════════════════
document.getElementById('patient-form').addEventListener('submit', async (e) => {
  e.preventDefault();

  const btn = document.getElementById('btn-predict');
  btn.disabled = true;
  btn.textContent = '⏳ Procesando…';

  // Construir payload con los 16 campos del formulario
  const payload = {};
  for (const id of FIELD_IDS) {
    const el = document.getElementById(id);
    payload[id] = parseInt(el.value, 10);
  }

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Error HTTP ${response.status}`);
    }

    const data = await response.json();
    showResult(data);

  } catch (error) {
    alert(`❌ Error al obtener la predicción:\n\n${error.message}\n\nVerifica que el backend esté ejecutándose en ${API_URL}.`);
  } finally {
    btn.disabled = false;
    btn.textContent = '🔍 Predecir riesgo de readmisión';
  }
});

// ═════════════════════════════════════════════════════════════
// 3. MOSTRAR RESULTADO
// ═════════════════════════════════════════════════════════════
function showResult(data) {
  const prob    = data.probability;
  const probPct = Math.round(prob * 100);

  // Clasificar nivel de riesgo
  let level, icon, label;
  if (prob >= T_MOD_MAX) {
    level = 'high'; icon = '🔴'; label = 'Riesgo ALTO de readmisión';
  } else if (prob >= T_LOW_MAX) {
    level = 'medium'; icon = '🟠'; label = 'Riesgo MODERADO de readmisión';
  } else {
    level = 'low'; icon = '🟢'; label = 'Riesgo BAJO de readmisión';
  }

  // Indicador de riesgo
  const indicator = document.getElementById('risk-indicator');
  indicator.className = `risk-indicator ${level}`;
  document.getElementById('risk-icon').textContent = icon;
  document.getElementById('risk-text').textContent = label;
  document.getElementById('prob-text').textContent = `${probPct} %`;

  // Caja de acción recomendada
  const actionBox = document.getElementById('risk-action-box');
  actionBox.className = `risk-action-box ${level}`;
  actionBox.textContent = `💡 ${data.recommended_action}`;

  // Barra de probabilidad (animación con leve retraso)
  const fill = document.getElementById('prob-bar-fill');
  fill.style.width = '0%';
  setTimeout(() => { fill.style.width = `${probPct}%`; }, 80);

  // Gráfico SHAP — Top 8 variables por valor absoluto
  renderSHAP(data.shap_values, data.features);

  // Timestamp legible
  const ts = new Date(data.timestamp).toLocaleString('es-CL', {
    dateStyle: 'medium', timeStyle: 'medium'
  });
  document.getElementById('timestamp-text').textContent =
    `Predicción generada el ${ts}`;

  // Mostrar resultado, ocultar formulario
  document.getElementById('form-section').style.display   = 'none';
  document.getElementById('result-section').style.display = 'block';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ═════════════════════════════════════════════════════════════
// 4. RENDERIZADO DEL GRÁFICO SHAP
// ═════════════════════════════════════════════════════════════
function renderSHAP(values, labels) {
  const chart = document.getElementById('shap-chart');
  chart.innerHTML = '';

  // Ordenar por |valor| descendente y tomar los 8 primeros
  const indexed = values.map((v, i) => ({ v, label: labels[i] }));
  indexed.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const top8 = indexed.slice(0, 8);

  const maxAbs = Math.max(...top8.map(x => Math.abs(x.v)), 0.001);

  for (const item of top8) {
    const widthPct = Math.min((Math.abs(item.v) / maxAbs) * 100, 100);
    const positive = item.v >= 0;
    const sign     = positive ? '+' : '−';

    const row = document.createElement('div');
    row.className = 'shap-row';
    row.innerHTML = `
      <span class="shap-label" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</span>
      <div class="shap-bar-wrap">
        <div class="shap-bar ${positive ? 'positive' : 'negative'}" style="width:0%">
          <span class="shap-val">${sign}${Math.abs(item.v).toFixed(4)}</span>
        </div>
      </div>
    `;
    chart.appendChild(row);
  }

  // Animar las barras tras un breve retraso
  setTimeout(() => {
    chart.querySelectorAll('.shap-bar').forEach((bar, idx) => {
      const item = top8[idx];
      const widthPct = Math.min((Math.abs(item.v) / maxAbs) * 100, 100);
      bar.style.width = `${widthPct}%`;
    });
  }, 100);
}

// ═════════════════════════════════════════════════════════════
// 5. RESET DEL FORMULARIO
// ═════════════════════════════════════════════════════════════
function resetForm() {
  document.getElementById('result-section').style.display = 'none';
  document.getElementById('form-section').style.display   = 'block';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ═════════════════════════════════════════════════════════════
// 6. PANEL DE MONITOREO — invoca /monitor
// ═════════════════════════════════════════════════════════════
async function loadMonitor() {
  const div = document.getElementById('monitor-content');
  div.innerHTML = "<p class='muted'>Cargando estadísticas…</p>";

  try {
    const response = await fetch(`${API_URL}/monitor`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();

    if (!data.total_predictions || data.total_predictions === 0) {
      div.innerHTML = "<p class='muted'>No hay predicciones registradas aún. Genera la primera con el formulario.</p>";
      return;
    }

    const dist = data.risk_level_distribution || {
      BAJO:     { count: 0, pct: 0 },
      MODERADO: { count: 0, pct: 0 },
      ALTO:     { count: 0, pct: 0 },
    };

    div.innerHTML = `
      <div class="monitor-grid">
        <div class="monitor-stat">
          <div class="val">${data.total_predictions.toLocaleString('es-CL')}</div>
          <div class="lbl">Predicciones totales</div>
        </div>
        <div class="monitor-stat">
          <div class="val">${formatProb(data.avg_probability)}</div>
          <div class="lbl">Probabilidad media</div>
        </div>
        <div class="monitor-stat">
          <div class="val">${formatProb(data.min_probability)}</div>
          <div class="lbl">Probabilidad mínima</div>
        </div>
        <div class="monitor-stat">
          <div class="val">${formatProb(data.max_probability)}</div>
          <div class="lbl">Probabilidad máxima</div>
        </div>
        <div class="monitor-stat bajo">
          <div class="val">${dist.BAJO.pct} %</div>
          <div class="lbl">Distribución BAJO</div>
        </div>
        <div class="monitor-stat moderado">
          <div class="val">${dist.MODERADO.pct} %</div>
          <div class="lbl">Distribución MODERADO</div>
        </div>
        <div class="monitor-stat alto">
          <div class="val">${dist.ALTO.pct} %</div>
          <div class="lbl">Distribución ALTO</div>
        </div>
        <div class="monitor-stat">
          <div class="val" style="font-size:0.85rem">${formatTimestamp(data.last_prediction_ts)}</div>
          <div class="lbl">Última predicción</div>
        </div>
      </div>
      ${monitorAlertHtml(dist)}
    `;
  } catch (error) {
    div.innerHTML = `<p class='muted'>⚠️ Error al cargar estadísticas: ${escapeHtml(error.message)}</p>`;
  }
}

// Genera alerta de deriva conceptual si la distribución difiere de la esperada
// Baseline esperada del entrenamiento: BAJO 21%, MODERADO 46%, ALTO 33%
function monitorAlertHtml(dist) {
  const baseline = { BAJO: 21, MODERADO: 46, ALTO: 33 };
  const tolerance = 10; // puntos porcentuales
  const drifts = [];
  for (const lvl of ['BAJO', 'MODERADO', 'ALTO']) {
    const current = dist[lvl].pct;
    const diff = Math.abs(current - baseline[lvl]);
    if (diff > tolerance) {
      drifts.push(`${lvl}: ${current}% (esperado ${baseline[lvl]}%, Δ=${diff.toFixed(1)})`);
    }
  }
  if (drifts.length === 0) {
    return `<p style="margin-top:14px;font-size:0.82rem;color:var(--success);">✓ Distribución dentro del rango esperado. No se detecta deriva conceptual.</p>`;
  }
  return `
    <div style="margin-top:14px; padding:10px 14px; background:#FEF3C7; border-left:4px solid var(--warning); border-radius:4px; font-size:0.82rem; color:#92400E;">
      ⚠️ <b>Posible deriva conceptual detectada</b><br>
      ${drifts.join('<br>')}<br>
      <span style="font-style:italic">Considera revisar los datos recientes y reentrenar el modelo si la desviación persiste.</span>
    </div>
  `;
}

// ═════════════════════════════════════════════════════════════
// Utilidades
// ═════════════════════════════════════════════════════════════
function formatProb(v) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  return Number(v).toFixed(4);
}

function formatTimestamp(ts) {
  if (!ts || ts === 'N/A') return '—';
  try {
    return new Date(ts).toLocaleString('es-CL', {
      day: '2-digit', month: '2-digit', year: '2-digit',
      hour: '2-digit', minute: '2-digit'
    });
  } catch (e) {
    return ts;
  }
}

function escapeHtml(s) {
  if (typeof s !== 'string') return '';
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
