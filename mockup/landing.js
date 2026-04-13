/* ═══════════════════════════════════════════════════════════════
   ReadmitAI — Landing Page JavaScript
   Predicción de Readmisión Hospitalaria · ACIF104 UNAB 2025
   ═══════════════════════════════════════════════════════════════ */

// ── Configuración ──────────────────────────────────────────────
const API_URL = "http://localhost:8000";

const FEATURE_LABELS = [
  "Días en hospital", "Proced. laboratorio", "Proced. clínicos",
  "Medicamentos", "Visitas ambulatorias", "Ingresos previos", "Visitas urgencias",
  "Edad (codificada)", "Test glucosa", "Test HbA1c", "Cambio medicamento",
  "Med. diabetes", "Especialidad médica", "Diagnóstico 1",
  "Diagnóstico 2", "Diagnóstico 3",
];

const FIELD_IDS = [
  "time_in_hospital", "n_lab_procedures", "n_procedures", "n_medications",
  "n_outpatient", "n_inpatient", "n_emergency", "age_enc",
  "glucose_test_enc", "A1Ctest_enc", "change_enc", "diabetes_med_enc",
  "medical_specialty_enc", "diag_1_enc", "diag_2_enc", "diag_3_enc",
];

// ── Navbar toggle (mobile) ─────────────────────────────────────
document.getElementById("nav-toggle").addEventListener("click", () => {
  document.getElementById("nav-links").classList.toggle("open");
});

// Cerrar menú al hacer clic en un enlace
document.querySelectorAll(".nav-links a").forEach((link) => {
  link.addEventListener("click", () => {
    document.getElementById("nav-links").classList.remove("open");
  });
});

// ── Scroll animations ──────────────────────────────────────────
function setupScrollAnimations() {
  const targets = document.querySelectorAll(
    ".team-card, .pipeline-step, .feature-card, .predict-card, .tech-stack"
  );
  targets.forEach((el) => el.classList.add("animate-in"));

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
        }
      });
    },
    { threshold: 0.15 }
  );

  targets.forEach((el) => observer.observe(el));
}

// ── Navbar background on scroll ────────────────────────────────
function setupNavScroll() {
  const navbar = document.getElementById("navbar");
  window.addEventListener("scroll", () => {
    if (window.scrollY > 50) {
      navbar.style.background = "rgba(15,43,78,.98)";
    } else {
      navbar.style.background = "rgba(15,43,78,.92)";
    }
  });
}

// ── Datos de prueba (demo) ──────────────────────────────────────
const MOCK_DATA = {
  probability: 0.72,
  risk_label: "Se recomienda seguimiento intensivo y plan de egreso personalizado.",
  shap_values: [0.085, 0.042, -0.018, 0.063, -0.005, 0.192, 0.031, 0.055, -0.012, 0.034, 0.027, 0.019, -0.041, 0.068, -0.023, 0.015],
  features: FEATURE_LABELS,
  timestamp: new Date().toLocaleString("es-CL"),
};

// ── Form submission (modo demo con datos de prueba) ────────────
document.getElementById("patient-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const btn = document.getElementById("btn-predict");
  const resultPanel = document.getElementById("result-panel");

  // Si el resultado ya está visible, ocultarlo y volver al formulario
  if (resultPanel.style.display === "block") {
    resetPrediction();
    return;
  }

  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Procesando…';

  // Simular un pequeño delay para la experiencia de usuario
  await new Promise((resolve) => setTimeout(resolve, 800));

  // Usar datos de prueba con timestamp actualizado
  const demoData = { ...MOCK_DATA, timestamp: new Date().toLocaleString("es-CL") };
  showResult(demoData);

  btn.disabled = false;
  btn.innerHTML = '<span class="btn-icon">🔍</span> Predecir riesgo de readmisión';
});

// ── Show result ────────────────────────────────────────────────
function showResult(data) {
  const prob = data.probability;
  const probPct = Math.round(prob * 100);

  // Risk banner
  const banner = document.getElementById("risk-banner");
  const icon = document.getElementById("risk-icon");
  const level = document.getElementById("risk-level");
  const probEl = document.getElementById("risk-prob");

  banner.className = "risk-banner";
  if (prob >= 0.6) {
    banner.classList.add("high");
    icon.textContent = "🔴";
    level.textContent = "Riesgo ALTO de readmisión";
  } else if (prob >= 0.4) {
    banner.classList.add("medium");
    icon.textContent = "🟠";
    level.textContent = "Riesgo MODERADO de readmisión";
  } else {
    banner.classList.add("low");
    icon.textContent = "🟢";
    level.textContent = "Riesgo BAJO de readmisión";
  }
  probEl.textContent = `${probPct} %`;

  // Advice
  const advice = document.getElementById("risk-advice");
  advice.textContent = data.risk_label
    ? `💡 ${data.risk_label}`
    : prob >= 0.6
    ? "💡 Se recomienda seguimiento intensivo y plan de egreso personalizado."
    : prob >= 0.4
    ? "💡 Considerar seguimiento ambulatorio y revisión de medicación."
    : "💡 Bajo riesgo. Mantener controles habituales.";

  // Probability bar
  const fill = document.getElementById("prob-fill");
  const pctText = document.getElementById("prob-pct-text");
  fill.style.width = "0%";
  pctText.textContent = `${probPct} %`;
  requestAnimationFrame(() => {
    fill.style.width = `${probPct}%`;
  });

  // SHAP chart
  if (data.shap_values && data.features) {
    renderSHAP(data.shap_values, data.features);
  } else {
    document.getElementById("shap-chart").innerHTML =
      '<p style="color:var(--text-muted);font-size:.88rem;">Datos SHAP no disponibles en esta respuesta.</p>';
  }

  // Timestamp
  const ts = document.getElementById("result-timestamp");
  ts.textContent = data.timestamp
    ? `Predicción generada: ${data.timestamp}`
    : `Predicción generada: ${new Date().toLocaleString("es-CL")}`;

  // Toggle panels
  document.getElementById("form-panel").style.display = "none";
  document.getElementById("result-panel").style.display = "block";
  document.getElementById("result-panel").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Show error ─────────────────────────────────────────────────
function showError(message) {
  const resultPanel = document.getElementById("result-panel");
  const formPanel = document.getElementById("form-panel");

  document.getElementById("risk-banner").className = "risk-banner high";
  document.getElementById("risk-icon").textContent = "⚠️";
  document.getElementById("risk-level").textContent = "Error de conexión";
  document.getElementById("risk-prob").textContent = "—";
  document.getElementById("risk-advice").textContent =
    `No se pudo conectar con el servidor: ${message}. Asegúrate de que el backend esté ejecutándose en ${API_URL}`;
  document.getElementById("prob-fill").style.width = "0%";
  document.getElementById("prob-pct-text").textContent = "— %";
  document.getElementById("shap-chart").innerHTML = "";
  document.getElementById("result-timestamp").textContent = "";

  formPanel.style.display = "none";
  resultPanel.style.display = "block";
  resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Render SHAP chart ──────────────────────────────────────────
function renderSHAP(values, labels) {
  const chart = document.getElementById("shap-chart");
  chart.innerHTML = "";

  const indexed = values.map((v, i) => ({ v, label: labels[i] }));
  indexed.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const top8 = indexed.slice(0, 8);
  const maxAbs = Math.max(...top8.map((x) => Math.abs(x.v)), 0.001);

  for (const item of top8) {
    const pct = Math.min((Math.abs(item.v) / maxAbs) * 100, 100);
    const positive = item.v >= 0;
    const sign = positive ? "+" : "−";

    const row = document.createElement("div");
    row.className = "shap-row";
    row.innerHTML = `
      <span class="shap-label">${item.label}</span>
      <div class="shap-bar-wrap">
        <div class="shap-bar ${positive ? "positive" : "negative"}" style="width:${pct}%">
          <span class="shap-val">${sign}${Math.abs(item.v).toFixed(4)}</span>
        </div>
      </div>
    `;
    chart.appendChild(row);
  }
}

// ── Reset prediction ───────────────────────────────────────────
function resetPrediction() {
  document.getElementById("result-panel").style.display = "none";
  document.getElementById("form-panel").style.display = "block";
  document.getElementById("form-panel").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Init ───────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  setupScrollAnimations();
  setupNavScroll();
});
