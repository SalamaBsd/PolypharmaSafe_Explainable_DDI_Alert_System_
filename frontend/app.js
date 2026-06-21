// ClearDDI — front-end logic
// Prototype académique d'aide à la décision — non destiné à un usage clinique réel.

const API = "";

const $ = (sel) => document.querySelector(sel);

const drugAInput = $("#drugA");
const drugBInput = $("#drugB");
const analyzeBtn = $("#analyzeBtn");
const swapBtn = $("#swapBtn");
const errorBanner = $("#errorBanner");
const resultsEl = $("#results");
const suggestionsA = $("#suggestionsA");
const suggestionsB = $("#suggestionsB");

// ── Autocomplétion légère (suggestion uniquement, ne bloque rien) ───────
function setupAutocomplete(input, box) {
  let debounceTimer = null;
  let activeIndex = -1;

  input.addEventListener("input", () => {
    const q = input.value.trim();
    clearTimeout(debounceTimer);
    if (q.length < 2) {
      box.classList.remove("open");
      box.innerHTML = "";
      return;
    }
    debounceTimer = setTimeout(async () => {
      try {
        const res = await fetch(`${API}/api/search?q=${encodeURIComponent(q)}`);
        if (!res.ok) return;
        const data = await res.json();
        renderSuggestions(data.matches || []);
      } catch (e) { /* silencieux : l'autocomplétion est un confort, pas une dépendance */ }
    }, 180);
  });

  function renderSuggestions(matches) {
    activeIndex = -1;
    if (!matches.length) {
      box.classList.remove("open");
      box.innerHTML = "";
      return;
    }
    box.innerHTML = matches
      .map((m, i) => `<div class="suggestion-item" data-idx="${i}">${capitalize(m)}</div>`)
      .join("");
    box.classList.add("open");
    box.querySelectorAll(".suggestion-item").forEach((el) => {
      el.addEventListener("mousedown", (e) => {
        e.preventDefault();
        input.value = capitalize(el.textContent);
        box.classList.remove("open");
        box.innerHTML = "";
      });
    });
  }

  input.addEventListener("blur", () => {
    setTimeout(() => box.classList.remove("open"), 100);
  });
}

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

setupAutocomplete(drugAInput, suggestionsA);
setupAutocomplete(drugBInput, suggestionsB);

// ── Exemples cliquables ──────────────────────────────────────────────────
document.querySelectorAll(".sample-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    drugAInput.value = capitalize(chip.dataset.a);
    drugBInput.value = capitalize(chip.dataset.b);
    runAnalysis();
  });
});

// ── Inversion A/B ─────────────────────────────────────────────────────
swapBtn.addEventListener("click", () => {
  const tmp = drugAInput.value;
  drugAInput.value = drugBInput.value;
  drugBInput.value = tmp;
});

// ── Lancement de l'analyse ────────────────────────────────────────────
analyzeBtn.addEventListener("click", runAnalysis);
[drugAInput, drugBInput].forEach((inp) =>
  inp.addEventListener("keydown", (e) => {
    if (e.key === "Enter") runAnalysis();
  })
);

async function runAnalysis() {
  const a = drugAInput.value.trim();
  const b = drugBInput.value.trim();

  hideError();

  if (!a || !b) {
    showError("Veuillez saisir le nom des deux médicaments.");
    return;
  }

  setLoading(true);
  resultsEl.hidden = true;

  try {
    const res = await fetch(`${API}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ drug_a: a, drug_b: b }),
    });

    const data = await res.json();

    if (res.status === 404 && data.error === "not_found") {
      const which = data.which === "A" ? a : b;
      showError(`Le médicament « ${which} » n'a pas été trouvé dans la base de référence.`);
      return;
    }
    if (!res.ok) {
      showError("Une erreur est survenue pendant l'analyse. Veuillez réessayer.");
      return;
    }

    renderResults(data);
  } catch (e) {
    showError("Impossible de contacter le serveur d'analyse.");
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  analyzeBtn.querySelector(".btn-label").textContent = isLoading ? "Analyse en cours…" : "Lancer l'analyse";
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.hidden = false;
}
function hideError() {
  errorBanner.hidden = true;
}

// ── Rendu des résultats ──────────────────────────────────────────────
function riskClass(level) {
  return level === "élevé" ? "eleve" : level === "modéré" ? "modere" : "low";
}

function renderResults(data) {
  const { drug_a, drug_b, prediction, clinical_type, side_effects, fragments, reasons, model_metrics } = data;

  const rc = riskClass(prediction.risk_level);
  const probaPct = Math.round(prediction.probability * 100);
  const confPct = Math.round(prediction.confidence * 100);

  const gaugeCircumference = 2 * Math.PI * 70;
  const gaugeOffset = gaugeCircumference * (1 - prediction.probability);
  const gaugeColor = rc === "eleve" ? "var(--risk-high)" : rc === "modere" ? "var(--risk-mod)" : "var(--risk-low)";

  const headline = prediction.interaction_predicted
    ? `Interaction <em class="${rc}">${clinical_type.label.toLowerCase()}</em> probable entre ces deux substances.`
    : `Aucune interaction significative détectée par le modèle entre ces deux substances.`;

  resultsEl.innerHTML = `
    <div class="print-header">
      <img src="/assets/logo.png" alt="ClearDDI" class="print-logo">
      <div class="print-header-text">
        <h1>Fiche d'analyse — Interaction médicamenteuse</h1>
        <p class="print-meta">Générée le ${new Date().toLocaleDateString("fr-FR")} à ${new Date().toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" })} · ClearDDI (prototype académique)</p>
      </div>
      <div class="print-pair">
        <span class="print-drug">${escapeHtml(drug_a.name)}</span>
        <span class="print-x">×</span>
        <span class="print-drug">${escapeHtml(drug_b.name)}</span>
      </div>
      <span class="print-risk-badge risk-${rc}">Risque ${prediction.risk_level} — probabilité ${probaPct}%</span>
    </div>

    <section>
      <h2 class="section-title">Résultat de la prédiction</h2>
      <div class="result-hero risk-${rc}">
        <div class="gauge-col">
          <svg width="160" height="160" viewBox="0 0 160 160">
            <circle cx="80" cy="80" r="70" fill="none" stroke="#E1E8E8" stroke-width="13"/>
            <circle cx="80" cy="80" r="70" fill="none" stroke="${gaugeColor}" stroke-width="13"
              stroke-dasharray="${gaugeCircumference}" stroke-dashoffset="${gaugeOffset}" stroke-linecap="round"
              transform="rotate(-90 80 80)"/>
            <text x="80" y="76" text-anchor="middle" font-family="JetBrains Mono" font-size="28" font-weight="700" fill="#1B3A4B">${probaPct}%</text>
            <text x="80" y="96" text-anchor="middle" font-family="Inter" font-size="10.5" fill="#5B6E70">probabilité</text>
          </svg>
          <span class="gauge-label">Score de risque</span>
        </div>
        <div class="result-main">
          <div class="result-pair">
            <span class="drug-pill">${escapeHtml(drug_a.name)}</span>
            <span class="x">×</span>
            <span class="drug-pill">${escapeHtml(drug_b.name)}</span>
          </div>
          <span class="risk-badge ${rc}"><span class="dot"></span> Risque ${prediction.risk_level}</span>
          <h3 class="result-headline">${headline}</h3>
          <div class="confidence-row">
            <span class="confidence-label">Score de confiance du modèle</span>
            <div class="confidence-bar"><div class="confidence-fill" style="width:${confPct}%"></div></div>
            <span class="confidence-value">${prediction.confidence.toFixed(2)}</span>
          </div>
          <div class="mol-row">
            ${moleculeCardA(drug_a)}
            ${moleculeCardB(drug_b)}
          </div>
        </div>
      </div>
    </section>

    <section>
      <h2 class="section-title">Explicabilité</h2>
      <div class="grid-2">
        <div class="card">
          <h3>Raisons possibles de l'alerte</h3>
          ${reasons.map(reasonItem).join("")}
        </div>
        <div class="card">
          <h3>Fragments influents (SHAP)</h3>
          ${fragments.slice(0, 6).map(featRow).join("")}
          <div class="mol-note">
            <strong>Lecture :</strong> chaque barre représente la contribution d'un bit de fingerprint moléculaire (Morgan, rayon 2) à la prédiction. Le rouge augmente le risque prédit, le teal le réduit.
          </div>
          ${fragmentChips(fragments)}
        </div>
      </div>
    </section>

    <section>
      <div class="grid-2">
        <div class="card">
          <h3>Effets secondaires potentiels associés</h3>
          <div class="se-tags">
            ${side_effects.map((s) => `<span class="se-tag">${escapeHtml(s)}</span>`).join("")}
          </div>
        </div>
        <div class="card">
          <h3>Type clinique le plus probable</h3>
          ${clinical_type.ranking.map(rankRow).join("")}
          <div class="mol-note">
            Classement prédit par un second modèle (XGBoost multi-classe) entraîné sur le mécanisme clinique des interactions documentées.
          </div>
        </div>
      </div>
    </section>

    <section>
      <h2 class="section-title">Fiche synthétique pharmacien</h2>
      <div class="sheet-card">
        <div class="sheet-header">
          <div>
            <h3>Fiche d'alerte — ${escapeHtml(drug_a.name)} × ${escapeHtml(drug_b.name)}</h3>
            <span class="sub">Générée le ${new Date().toLocaleDateString("fr-FR")} · Modèle XGBoost (AUC cold-start ${model_metrics.cold_start_auc?.toFixed(3) ?? "—"})</span>
          </div>
          <button class="export-btn" onclick="window.print()" type="button">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 3v12m0 0 4-4m-4 4-4-4M5 21h14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
            Imprimer / exporter
          </button>
        </div>
        <div class="sheet-grid">
          <div class="sheet-stat">
            <div class="label">Niveau de risque</div>
            <div class="value risk-level ${rc}">${prediction.risk_level}</div>
          </div>
          <div class="sheet-stat">
            <div class="label">Probabilité</div>
            <div class="value">${prediction.probability.toFixed(2)}</div>
          </div>
          <div class="sheet-stat">
            <div class="label">Confiance modèle</div>
            <div class="value">${prediction.confidence.toFixed(2)}</div>
          </div>
          <div class="sheet-stat">
            <div class="label">Type clinique</div>
            <div class="value" style="font-size:14px;">${escapeHtml(clinical_type.label)}</div>
          </div>
        </div>
      </div>
    </section>

    <div class="print-footer">
      <p><strong>Prototype académique d'aide à la décision.</strong> Cet outil ne constitue pas un dispositif médical validé et ne doit pas remplacer le jugement clinique d'un professionnel de santé. Document généré automatiquement par ClearDDI à des fins pédagogiques.</p>
    </div>
  `;

  resultsEl.hidden = false;
  resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });

  // Charger les structures moléculaires réelles de façon asynchrone (SVG RDKit)
  loadMoleculeSvg(drug_a.name, "mol-svg-a");
  loadMoleculeSvg(drug_b.name, "mol-svg-b");
}

function moleculeCardA(drug) {
  return `<div class="mol-card"><div id="mol-svg-a">…</div><div class="mol-label">${escapeHtml(drug.name)} (${drug.drugbank_id})</div></div>`;
}
function moleculeCardB(drug) {
  return `<div class="mol-card"><div id="mol-svg-b">…</div><div class="mol-label">${escapeHtml(drug.name)} (${drug.drugbank_id})</div></div>`;
}

async function loadMoleculeSvg(name, targetId) {
  try {
    const res = await fetch(`${API}/api/molecule_svg?name=${encodeURIComponent(name)}`);
    if (!res.ok) return;
    const data = await res.json();
    const el = document.getElementById(targetId);
    if (el && data.svg) el.innerHTML = data.svg;
  } catch (e) { /* silencieux */ }
}

function reasonItem(r) {
  return `
    <div class="reason-item">
      <div class="reason-icon">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.8"/><path d="M9 12l2 2 4-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </div>
      <div class="reason-text">
        <div class="t">${escapeHtml(r.title)}</div>
        <div class="d">${escapeHtml(r.detail)}</div>
      </div>
    </div>`;
}

function featRow(f) {
  const pct = Math.min(100, Math.abs(f.shap_value) * 80);
  const cls = f.shap_value > 0 ? "pos" : "neg";
  const sign = f.shap_value > 0 ? "+" : "";
  const color = f.shap_value > 0 ? "var(--risk-high)" : "var(--teal)";
  const name = `bit_${f.bit}`;
  return `
    <div class="feat-row">
      <span class="feat-name">${name}</span>
      <div class="feat-track"><div class="feat-fill ${cls}" style="width:${pct}%"></div></div>
      <span class="feat-val" style="color:${color};">${sign}${f.shap_value.toFixed(2)}</span>
    </div>`;
}

function fragmentChips(fragments) {
  const withFrag = fragments.filter((f) => f.fragment_smiles);
  if (!withFrag.length) return "";
  return `<div class="frag-chip-row">${withFrag
    .map((f) => `<span class="frag-chip" title="bit ${f.bit}, médicament ${f.source_drug}">${escapeHtml(f.fragment_smiles)}</span>`)
    .join("")}</div>`;
}

function rankRow(r) {
  const pct = Math.round(r.probability * 100);
  return `
    <div class="rank-row">
      <span class="rank-name">${escapeHtml(r.type)}</span>
      <div class="rank-track"><div class="rank-fill" style="width:${pct}%"></div></div>
      <span class="rank-val">${pct}%</span>
    </div>`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}
