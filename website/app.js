const state = {
  entities: [],
  normalized: [],
  byName: new Map(),
  degrees: [],
  maxDegree: 1,
  meta: null,
  lastDatasetResults: [],
};

const STOPWORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is", "it",
  "of", "on", "or", "that", "the", "this", "to", "was", "were", "what", "when", "where", "which",
  "who", "with", "its", "their", "our", "your",
]);

const normalize = (value) => value
  .normalize("NFKD")
  .replace(/[\u0300-\u036f]/g, "")
  .toLowerCase()
  .replace(/[^\p{L}\p{N}]+/gu, " ")
  .trim()
  .replace(/\s+/g, " ");

function tierFor(degree) {
  const superHubThreshold = state.meta?.super_hub_threshold_p99 ?? 35;
  const hubThreshold = state.meta?.hub_threshold_p95 ?? 6;
  if (degree >= superHubThreshold) return ["Super hub", "super-hub"];
  if (degree >= hubThreshold) return ["Hub · top ≈5%", "hub"];
  if (degree >= 2) return ["Connected", "connected"];
  if (degree === 1) return ["Tail", "tail"];
  return ["Unreferenced", "tail"];
}

function upperBound(sorted, target) {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (sorted[mid] <= target) low = mid + 1;
    else high = mid;
  }
  return low;
}

function rankCopy(degree) {
  if (degree <= 0 || !state.degrees.length) return "No incoming forward facts";
  const greater = state.degrees.length - upperBound(state.degrees, degree);
  const pct = (greater / state.degrees.length) * 100;
  if (pct < 0.01) return "Highest-connectivity tier";
  if (pct < 1) return `Only ${pct.toFixed(2)}% rank higher`;
  return `Only ${pct.toFixed(1)}% rank higher`;
}

function ringScore(degree) {
  return Math.max(3, (Math.log1p(degree) / Math.log1p(state.maxDegree)) * 100);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

function entityCard(record, count = null) {
  const [name, qid, degree] = record;
  const [tierName, tierClass] = tierFor(degree);
  const qidLink = qid && qid.startsWith("Q")
    ? `<a href="https://www.wikidata.org/wiki/${encodeURIComponent(qid)}" target="_blank" rel="noreferrer">${escapeHtml(qid)} ↗</a>`
    : "No resolved QID";
  const mention = count === null ? "" : `<span>${count} matched row${count === 1 ? "" : "s"}</span>`;
  return `
    <article class="entity-result">
      <div class="degree-ring" style="--score:${ringScore(degree).toFixed(1)}%"><span>${degree >= 1000 ? `${(degree / 1000).toFixed(1)}k` : degree}</span></div>
      <div>
        <strong class="entity-name">${escapeHtml(name)}</strong>
        <div class="entity-meta"><span class="tier tier-${tierClass}">${tierName}</span><span>${rankCopy(degree)}</span>${mention}${qidLink}</div>
      </div>
      <div class="entity-degree"><strong>${degree.toLocaleString()}</strong><span>object in-degree</span></div>
    </article>`;
}

function findEntities(text, allowSuggestions = true) {
  const clean = normalize(text);
  if (!clean) return { matches: [], suggestions: [] };
  const exact = state.byName.get(clean);
  if (exact) return { matches: exact.slice(0, 6), suggestions: [] };

  const sourceWords = String(text).match(/[\p{L}\p{N}]+/gu) || [];
  const words = sourceWords.map((word) => normalize(word));
  const occupied = new Set();
  const matches = [];
  const maxGram = Math.min(10, words.length);
  for (let size = maxGram; size >= 1; size -= 1) {
    for (let start = 0; start <= words.length - size; start += 1) {
      if (Array.from({ length: size }, (_, offset) => occupied.has(start + offset)).some(Boolean)) continue;
      const phrase = words.slice(start, start + size).join(" ");
      if (size === 1) {
        const isCapitalized = /^\p{Lu}/u.test(sourceWords[start] || "");
        if (phrase.length < 3 || STOPWORDS.has(phrase) || /^\d+$/.test(phrase) || (words.length > 1 && !isCapitalized)) continue;
      }
      const found = state.byName.get(phrase);
      if (!found) continue;
      found.slice(0, 2).forEach((record) => matches.push(record));
      for (let offset = 0; offset < size; offset += 1) occupied.add(start + offset);
    }
  }

  const unique = [...new Map(matches.map((record) => [`${record[0]}|${record[1]}`, record])).values()]
    .sort((a, b) => b[2] - a[2]);
  if (unique.length || !allowSuggestions || words.length > 7) return { matches: unique.slice(0, 12), suggestions: [] };

  const suggestions = [];
  for (let index = 0; index < state.entities.length; index += 1) {
    const candidate = state.normalized[index];
    if (candidate.includes(clean) || clean.includes(candidate)) {
      suggestions.push(state.entities[index]);
      if (suggestions.length >= 24) break;
    }
  }
  suggestions.sort((a, b) => Math.abs(normalize(a[0]).length - clean.length) - Math.abs(normalize(b[0]).length - clean.length));
  return { matches: [], suggestions: suggestions.slice(0, 5) };
}

function renderSentenceResult(text) {
  const container = document.querySelector("#sentence-results");
  const { matches, suggestions } = findEntities(text);
  if (matches.length) {
    container.innerHTML = `<p class="result-message">Found ${matches.length} matching ${matches.length === 1 ? "entity" : "entities"}. Popularity uses forward-edge object in-degree.</p><div class="entity-list">${matches.map((record) => entityCard(record)).join("")}</div>`;
    return;
  }
  const suggestionMarkup = suggestions.length
    ? `<div class="suggestions">${suggestions.map((record) => `<button type="button" data-suggestion="${escapeHtml(record[0])}">${escapeHtml(record[0])}</button>`).join("")}</div>`
    : "";
  container.innerHTML = `<div class="empty-state"><span class="empty-orbit" aria-hidden="true"></span><div><p><strong>No exact FactProp entity found.</strong><br>Try the canonical English entity name. Sentence matching currently uses graph labels rather than an external language model.</p>${suggestionMarkup}</div></div>`;
  container.querySelectorAll("[data-suggestion]").forEach((button) => button.addEventListener("click", () => {
    document.querySelector("#sentence-input").value = button.dataset.suggestion;
    renderSentenceResult(button.dataset.suggestion);
  }));
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { cell += '"'; i += 1; }
      else if (char === '"') quoted = false;
      else cell += char;
    } else if (char === '"') quoted = true;
    else if (char === ",") { row.push(cell); cell = ""; }
    else if (char === "\n") { row.push(cell); rows.push(row); row = []; cell = ""; }
    else if (char !== "\r") cell += char;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  return rows;
}

function collectJsonStrings(value, output, limit = 5000) {
  if (output.length >= limit) return;
  if (typeof value === "string") output.push(value);
  else if (Array.isArray(value)) value.forEach((item) => collectJsonStrings(item, output, limit));
  else if (value && typeof value === "object") Object.values(value).forEach((item) => collectJsonStrings(item, output, limit));
}

async function analyzeDataset(file) {
  const container = document.querySelector("#dataset-results");
  if (file.size > 5 * 1024 * 1024) {
    container.innerHTML = '<p class="result-message error">That file is larger than 5 MB. Please upload a smaller sample.</p>';
    return;
  }
  container.innerHTML = '<p class="result-message">Reading and analyzing locally…</p>';
  try {
    const text = await file.text();
    const extension = file.name.split(".").pop().toLowerCase();
    let items = [];
    if (extension === "json") {
      collectJsonStrings(JSON.parse(text), items);
    } else if (extension === "csv") {
      items = parseCsv(text).slice(0, 5000).map((row) => row.join(" "));
    } else {
      items = text.split(/\r?\n/).filter(Boolean).slice(0, 5000);
    }

    const aggregate = new Map();
    let matchedRows = 0;
    items.forEach((item) => {
      const { matches } = findEntities(item, false);
      if (matches.length) matchedRows += 1;
      matches.forEach((record) => {
        const key = `${record[0]}|${record[1]}`;
        const current = aggregate.get(key) || { record, count: 0 };
        current.count += 1;
        aggregate.set(key, current);
      });
    });
    const ranked = [...aggregate.values()].sort((a, b) => b.count - a.count || b.record[2] - a.record[2]);
    state.lastDatasetResults = ranked;
    const avg = ranked.length ? ranked.reduce((sum, entry) => sum + entry.record[2], 0) / ranked.length : 0;
    const list = ranked.length
      ? `<div class="entity-list">${ranked.slice(0, 30).map(({ record, count }) => entityCard(record, count)).join("")}</div>`
      : '<div class="empty-state"><span class="empty-orbit" aria-hidden="true"></span><p>No canonical FactProp entity labels were detected in this file.</p></div>';
    container.innerHTML = `
      <div class="dataset-summary">
        <div><strong>${items.length.toLocaleString()}</strong><span>rows / values</span></div>
        <div><strong>${matchedRows.toLocaleString()}</strong><span>matched rows</span></div>
        <div><strong>${Math.round(avg).toLocaleString()}</strong><span>mean in-degree</span></div>
      </div>
      <div class="dataset-actions"><p>${escapeHtml(file.name)} · ${ranked.length} unique entities</p>${ranked.length ? '<button id="download-results" type="button">Download CSV</button>' : ""}</div>
      ${list}`;
    document.querySelector("#download-results")?.addEventListener("click", downloadDatasetResults);
  } catch (error) {
    container.innerHTML = `<p class="result-message error">Could not read this file: ${escapeHtml(error.message)}</p>`;
  }
}

function downloadDatasetResults() {
  const rows = [["entity", "qid", "object_in_degree", "tier", "matched_rows"]];
  state.lastDatasetResults.forEach(({ record, count }) => rows.push([record[0], record[1] || "", record[2], tierFor(record[2])[0], count]));
  const csv = rows.map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(",")).join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "factprop-popularity-results.csv";
  anchor.click();
  URL.revokeObjectURL(url);
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => {
      const active = item === tab;
      item.classList.toggle("active", active);
      item.setAttribute("aria-selected", String(active));
      const panel = document.querySelector(`#${item.getAttribute("aria-controls")}`);
      panel.hidden = !active;
      panel.classList.toggle("active", active);
    });
  }));
}

function setupUploads() {
  const input = document.querySelector("#dataset-file");
  const drop = document.querySelector("#drop-zone");
  drop.addEventListener("click", () => input.click());
  input.addEventListener("change", () => input.files[0] && analyzeDataset(input.files[0]));
  ["dragenter", "dragover"].forEach((event) => drop.addEventListener(event, (e) => { e.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((event) => drop.addEventListener(event, (e) => { e.preventDefault(); drop.classList.remove("dragging"); }));
  drop.addEventListener("drop", (event) => event.dataTransfer.files[0] && analyzeDataset(event.dataTransfer.files[0]));
}

function setupCanvas() {
  const canvas = document.querySelector("#ripple-canvas");
  if (!canvas) return;
  const context = canvas.getContext("2d");
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let pointer = { x: .76, y: .24 };
  const nodes = [
    [.70,.19,4], [.78,.28,7], [.88,.21,3], [.83,.38,3], [.67,.35,2], [.92,.44,2],
    [.61,.27,2], [.74,.47,2], [.95,.31,2], [.86,.53,2], [.66,.52,2], [.96,.59,2],
  ];
  const links = [[0,1],[1,2],[1,3],[1,4],[3,5],[4,6],[3,7],[2,8],[5,9],[7,10],[9,11],[1,7]];
  function resize() {
    const box = canvas.getBoundingClientRect();
    const ratio = Math.min(devicePixelRatio || 1, 2);
    canvas.width = box.width * ratio; canvas.height = box.height * ratio;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
  }
  function draw(time = 0) {
    const { width, height } = canvas.getBoundingClientRect();
    context.clearRect(0, 0, width, height);
    context.lineWidth = 1;
    context.strokeStyle = "rgba(85,122,151,.18)";
    links.forEach(([a,b]) => {
      context.beginPath(); context.moveTo(nodes[a][0]*width,nodes[a][1]*height); context.lineTo(nodes[b][0]*width,nodes[b][1]*height); context.stroke();
    });
    const hubX = nodes[1][0]*width, hubY = nodes[1][1]*height;
    for (let r = 0; r < 3; r += 1) {
      const radius = 28 + ((reducedMotion ? r * 42 : (time * .035 + r * 54) % 160));
      context.beginPath(); context.arc(hubX, hubY, radius, 0, Math.PI*2);
      context.strokeStyle = `rgba(187,113,133,${Math.max(0,.22-radius/800)})`; context.stroke();
    }
    nodes.forEach(([x,y,size], index) => {
      const distance = Math.hypot(x-pointer.x,y-pointer.y);
      const pulse = index === 1 ? 2 : Math.max(0, .05-distance)*25;
      context.beginPath(); context.arc(x*width,y*height,size+pulse,0,Math.PI*2);
      context.fillStyle = index === 1 ? "rgba(187,113,133,.9)" : "rgba(85,122,151,.65)"; context.fill();
    });
    if (!reducedMotion) requestAnimationFrame(draw);
  }
  canvas.addEventListener("pointermove", (event) => {
    const box = canvas.getBoundingClientRect(); pointer = { x:(event.clientX-box.left)/box.width, y:(event.clientY-box.top)/box.height };
  });
  resize(); window.addEventListener("resize", resize); draw();
}

async function loadIndex() {
  const status = document.querySelector("#index-status");
  try {
    const response = await fetch("./data/entities.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    state.entities = payload.entities;
    state.meta = payload.meta;
    state.maxDegree = payload.meta.max_forward_in_degree;
    state.normalized = state.entities.map((record) => normalize(record[0]));
    state.entities.forEach((record, index) => {
      const key = state.normalized[index];
      const list = state.byName.get(key) || [];
      list.push(record);
      state.byName.set(key, list);
      if (record[2] > 0) state.degrees.push(record[2]);
    });
    state.degrees.sort((a, b) => a - b);
    status.classList.add("ready");
    status.innerHTML = `<i></i> ${payload.meta.entity_count.toLocaleString()} entities ready`;
    document.querySelector("#analyze-sentence").disabled = false;
  } catch (error) {
    status.classList.add("error");
    status.innerHTML = "<i></i> Graph index unavailable";
    document.querySelector("#sentence-results").innerHTML = `<p class="result-message error">The FactProp index could not load. ${escapeHtml(error.message)}</p>`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setupTabs(); setupUploads(); setupCanvas(); loadIndex();
  document.querySelector("#analyze-sentence").addEventListener("click", () => renderSentenceResult(document.querySelector("#sentence-input").value));
  document.querySelector("#sentence-input").addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") renderSentenceResult(event.currentTarget.value);
  });
  document.querySelectorAll("[data-example]").forEach((button) => button.addEventListener("click", () => {
    document.querySelector("#sentence-input").value = button.dataset.example;
    if (!document.querySelector("#analyze-sentence").disabled) renderSentenceResult(button.dataset.example);
  }));
  document.querySelector("#copy-bibtex")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    try {
      await navigator.clipboard.writeText(document.querySelector("#bibtex").innerText);
      button.textContent = "Copied";
      setTimeout(() => { button.textContent = "Copy BibTeX"; }, 1600);
    } catch {
      button.textContent = "Select to copy";
    }
  });
});
