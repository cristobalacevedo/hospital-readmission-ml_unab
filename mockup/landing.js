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
    navbar.classList.toggle("scrolled", window.scrollY > 50);
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
    icon.innerHTML = createSemaphore("high");
    level.textContent = "Riesgo ALTO de readmisión";
  } else if (prob >= 0.4) {
    banner.classList.add("medium");
    icon.innerHTML = createSemaphore("medium");
    level.textContent = "Riesgo MODERADO de readmisión";
  } else {
    banner.classList.add("low");
    icon.innerHTML = createSemaphore("low");
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
  document.getElementById("risk-icon").innerHTML = "⚠️";
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

// ── Render SHAP chart (top 5, interactivo) ────────────────────
function renderSHAP(values, labels) {
  const chart = document.getElementById("shap-chart");
  chart.innerHTML = "";
  chart.setAttribute("role", "list");
  chart.setAttribute("aria-label", "Top 5 factores más influyentes según SHAP");

  const indexed = values.map((v, i) => ({ v, label: labels[i] }));
  indexed.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const top5   = indexed.slice(0, 5);
  const maxAbs = Math.max(...top5.map((x) => Math.abs(x.v)), 0.001);

  for (const item of top5) {
    const pct      = Math.min((Math.abs(item.v) / maxAbs) * 100, 100);
    const positive = item.v >= 0;
    const sign     = positive ? "+" : "−";

    const row = document.createElement("div");
    row.className = "shap-row";
    row.setAttribute("role", "listitem");
    row.setAttribute("tabindex", "0");
    row.setAttribute("aria-label",
      `${item.label}: impacto ${sign}${Math.abs(item.v).toFixed(4)}, ${positive ? "aumenta" : "reduce"} el riesgo`);
    row.innerHTML = `
      <span class="shap-label">${item.label}</span>
      <div class="shap-bar-wrap">
        <div class="shap-bar ${positive ? "positive" : "negative"}" style="width:0%">
          <span class="shap-val">${sign}${Math.abs(item.v).toFixed(4)}</span>
        </div>
      </div>
    `;

    row.addEventListener("mousemove", (e) => showShapTooltip(e, item));
    row.addEventListener("mouseleave", hideShapTooltip);
    row.addEventListener("focus", () => {
      const rect = row.getBoundingClientRect();
      showShapTooltip({ clientX: rect.right, clientY: rect.top + rect.height / 2 }, item);
    });
    row.addEventListener("blur", hideShapTooltip);

    chart.appendChild(row);

    // Animar barra con doble rAF para activar transición CSS
    requestAnimationFrame(() => {
      const bar = row.querySelector(".shap-bar");
      requestAnimationFrame(() => { bar.style.width = `${pct}%`; });
    });
  }
}

// ── Reset prediction ───────────────────────────────────────────
function resetPrediction() {
  document.getElementById("result-panel").style.display = "none";
  document.getElementById("form-panel").style.display = "block";
  document.getElementById("form-panel").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Semáforo SVG ───────────────────────────────────────────────
function createSemaphore(level) {
  const r = level === "high"   ? "active" : "";
  const y = level === "medium" ? "active" : "";
  const g = level === "low"    ? "active" : "";
  const lbl = level === "high" ? "alto" : level === "medium" ? "moderado" : "bajo";
  return `<svg class="semaphore-svg" viewBox="0 0 56 110" width="44" height="87"
       aria-label="Semáforo: riesgo ${lbl}" role="img">
    <rect x="3" y="3" width="50" height="104" rx="10" fill="#1E293B" stroke="#334155" stroke-width="2"/>
    <circle cx="28" cy="22"  r="14" class="sem-light sem-red    ${r}"/>
    <circle cx="28" cy="55"  r="14" class="sem-light sem-yellow ${y}"/>
    <circle cx="28" cy="88"  r="14" class="sem-light sem-green  ${g}"/>
  </svg>`;
}

// ── SHAP Tooltip ───────────────────────────────────────────────
let shapTooltipEl = null;
function getShapTooltip() {
  if (!shapTooltipEl) shapTooltipEl = document.getElementById("shap-tooltip");
  return shapTooltipEl;
}

function showShapTooltip(e, item) {
  const positive = item.v >= 0;
  const sign     = positive ? "+" : "−";
  const impact   = positive ? "Aumenta el riesgo" : "Reduce el riesgo";
  const color    = positive ? "#F87171" : "#34D399";
  const t = getShapTooltip();
  t.innerHTML = `<strong>${item.label}</strong><br>
    Valor SHAP: ${sign}${Math.abs(item.v).toFixed(4)}<br>
    <span style="color:${color}">▶ ${impact}</span>`;
  t.classList.add("visible");
  t.removeAttribute("aria-hidden");
  positionShapTooltip(e);
}

function positionShapTooltip(e) {
  const t   = getShapTooltip();
  const gap = 14;
  let x = e.clientX + gap;
  let y = e.clientY - 10;
  if (x + 234 > window.innerWidth)  x = e.clientX - 234 - gap;
  if (y + 110 > window.innerHeight) y = e.clientY - 110;
  t.style.left = `${x}px`;
  t.style.top  = `${y}px`;
}

function hideShapTooltip() {
  const t = getShapTooltip();
  t.classList.remove("visible");
  t.setAttribute("aria-hidden", "true");
}

// ── Modo Oscuro ────────────────────────────────────────────────
function initDarkMode() {
  const btn  = document.getElementById("dark-toggle");
  const html = document.documentElement;
  const saved      = localStorage.getItem("readmitai-dark");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

  if (saved === "1" || (!saved && prefersDark)) {
    html.classList.add("dark");
    btn.textContent = "☀️";
    btn.setAttribute("aria-label", "Activar modo claro");
  }

  btn.addEventListener("click", () => {
    const isDark = html.classList.toggle("dark");
    btn.textContent = isDark ? "☀️" : "🌙";
    btn.setAttribute("aria-label", isDark ? "Activar modo claro" : "Activar modo oscuro");
    localStorage.setItem("readmitai-dark", isDark ? "1" : "0");
  });
}

// ── Monitor en Tiempo Real ─────────────────────────────────────
const MONITOR_THRESHOLD = 0.55;
const MONITOR_POLL_MS   = 30_000;
let   monitorIntervalId = null;

async function fetchMonitor() {
  const ctrl = new AbortController();
  const tid  = setTimeout(() => ctrl.abort(), 5000);
  try {
    const res = await fetch(`${API_URL}/monitor`, { signal: ctrl.signal });
    clearTimeout(tid);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch {
    clearTimeout(tid);
    // Fallback demo con variación aleatoria
    return {
      avg_probability:   Math.min(1, Math.max(0, 0.48 + (Math.random() - 0.5) * 0.06)),
      total_predictions: 142 + Math.floor(Math.random() * 8),
      drift_detected:    false,
    };
  }
}

function updateMonitorUI(data) {
  const avg     = data.avg_probability ?? 0;
  const isAlert = avg > MONITOR_THRESHOLD || !!data.drift_detected;

  document.getElementById("monitor-dot").className = `monitor-dot${isAlert ? " alert" : ""}`;

  const avgEl = document.getElementById("monitor-avg");
  avgEl.textContent = `${(avg * 100).toFixed(1)} %`;
  avgEl.className   = `monitor-value${isAlert ? " alert" : ""}`;

  document.getElementById("monitor-count").textContent = data.total_predictions ?? "—";

  const banner = document.getElementById("monitor-alert-banner");
  banner.style.display = isAlert ? "flex" : "none";

  document.getElementById("monitor-thresh-fill").style.width = `${Math.min(avg * 100, 100)}%`;
  document.getElementById("monitor-updated").textContent =
    `Actualizado: ${new Date().toLocaleTimeString("es-CL")}`;
}

async function pollMonitor() {
  try {
    updateMonitorUI(await fetchMonitor());
  } catch (err) {
    console.warn("Monitor: error inesperado", err);
  }
}

function initMonitor() {
  const header  = document.getElementById("monitor-header");
  const body    = document.getElementById("monitor-body");
  const chevron = document.getElementById("monitor-chevron");

  function toggle() {
    const open = body.classList.toggle("open");
    chevron.classList.toggle("open", open);
    header.classList.toggle("expanded", open);
    header.setAttribute("aria-expanded", String(open));
  }

  header.addEventListener("click", toggle);
  header.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
  });

  pollMonitor();
  monitorIntervalId = setInterval(pollMonitor, MONITOR_POLL_MS);
}

// ── Init ───────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  setupScrollAnimations();
  setupNavScroll();
  initDarkMode();
  initMonitor();
});
