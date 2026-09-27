// All the actual computer-vision work happens in Python (app.py calling your
// tested scripts). This file just wires up the tabs and asks the server for
// results, then drops the returned base64 images into <img> tags.

function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });
}

function renderMetrics(el, data) {
  if (data.sam2_missing) {
    el.innerHTML = `<p class="method-note">${data.sam2_missing}</p>`;
    return;
  }
  if (data.iou === undefined) { el.innerHTML = ""; return; }
  el.innerHTML = `
    <div class="metric"><div class="value">${data.iou.toFixed(3)}</div><div class="label">IoU (Jaccard)</div></div>
    <div class="metric"><div class="value">${data.dice.toFixed(3)}</div><div class="label">Dice (F1)</div></div>
    <div class="metric"><div class="value">${data.boundary_f.toFixed(3)}</div><div class="label">Boundary F-score (2px)</div></div>
  `;
}

function renderInfo(el, data) {
  const bits = [];
  if (data.bbox) bits.push(`bbox: [${data.bbox.join(", ")}]`);
  if (data.otsu_threshold !== undefined) bits.push(`Otsu threshold: ${data.otsu_threshold.toFixed(1)}`);
  if (data.polarity) bits.push(`polarity: ${data.polarity}`);
  el.textContent = bits.join("  ·  ");
}

async function loadModality(prefix, sampleId, modality) {
  const originalImg = document.getElementById(`${prefix}-original`);
  const boundaryImg = document.getElementById(`${prefix}-boundary`);
  const compareImg = document.getElementById(`${prefix}-compare`);
  const metricsEl = document.getElementById(`${prefix}-metrics`);
  const infoEl = document.getElementById(`${prefix}-info`);

  metricsEl.innerHTML = `<span class="metric-loading">Processing…</span>`;
  infoEl.textContent = "";

  try {
    const res = await fetch(`/api/process/${sampleId}/${modality}`);
    const data = await res.json();

    if (data.error) {
      metricsEl.innerHTML = `<p class="method-note">${data.error}</p>`;
      return;
    }

    originalImg.src = data.original;
    boundaryImg.src = data.boundary;
    if (data.compare) compareImg.src = data.compare;
    else compareImg.removeAttribute("src");

    renderInfo(infoEl, data);
    renderMetrics(metricsEl, data);
  } catch (err) {
    metricsEl.innerHTML = `<p class="method-note">Request failed: ${err.message}</p>`;
  }
}

function loadCurrentSample() {
  const sampleId = document.getElementById("sampleSelect").value;
  if (!sampleId) return;
  loadModality("q1", sampleId, "rgb");
  loadModality("q2", sampleId, "thermal");
}

function init() {
  initTabs();
  const select = document.getElementById("sampleSelect");
  if (select && select.options.length > 0) {
    select.addEventListener("change", loadCurrentSample);
    loadCurrentSample();
  }
}

document.addEventListener("DOMContentLoaded", init);
