// State
let allCases = [];
let currentFilteredList = [];
let currentIndexInFiltered = 0;
let currentCaseData = null;

// Pan & Zoom
let scale = 1.0;
let panX = 0;
let panY = 0;
let isDragging = false;
let startX = 0;
let startY = 0;

// Filter and UI elements
const viewport = document.getElementById("viewport");
const viewportContent = document.getElementById("viewportContent");
const radiographImg = document.getElementById("radiographImg");
const svgOverlay = document.getElementById("svgOverlay");

const btnChoiceA = document.getElementById("btnChoiceA");
const btnChoiceB = document.getElementById("btnChoiceB");
const btnChoiceU = document.getElementById("btnChoiceU");

const btnPrev = document.getElementById("btnPrev");
const btnNext = document.getElementById("btnNext");
const filterPriority = document.getElementById("filterPriority");
const progressText = document.getElementById("progressText");
const progressFill = document.getElementById("progressFill");

const contrastSlider = document.getElementById("contrastSlider");
const brightnessSlider = document.getElementById("brightnessSlider");
const btnInvert = document.getElementById("btnInvert");
let isInverted = false;

// Initialize
async function init() {
  setupEventListeners();
  await refreshQueue();
  if (currentFilteredList.length > 0) {
    loadCaseByIndex(currentFilteredList[0].queue_index);
  }
}

// Fetch queue and refresh stats
async function refreshQueue() {
  try {
    const res = await fetch("/api/queue");
    const data = await res.json();
    allCases = data.cases;

    // Update progress
    const reviewed = data.total_reviewed;
    const total = data.total_cases;
    const pct = total > 0 ? Math.round((reviewed / total) * 100) : 0;
    progressText.innerText = `${reviewed} / ${total} Reviewed (${pct}%)`;
    progressFill.style.width = `${pct}%`;

    applyFilter();
  } catch (err) {
    console.error("Failed to load queue:", err);
  }
}

function applyFilter() {
  const fVal = filterPriority.value;
  if (fVal === "all") {
    currentFilteredList = allCases;
  } else if (fVal === "unreviewed") {
    currentFilteredList = allCases.filter(c => !c.is_reviewed);
  } else if (fVal === "reviewed") {
    currentFilteredList = allCases.filter(c => c.is_reviewed);
  } else if (fVal === "1") {
    currentFilteredList = allCases.filter(c => c.priority === 1);
  } else if (fVal === "2") {
    currentFilteredList = allCases.filter(c => c.priority === 2);
  }
  currentIndexInFiltered = 0;
}

// Load specific case by 1-based queue_index
async function loadCaseByIndex(queueIdx) {
  try {
    const res = await fetch(`/api/case/${queueIdx}`);
    if (!res.ok) throw new Error("Failed to load case");
    const data = await res.json();
    currentCaseData = data;

    renderCaseDetails(data);
    renderRadiographAndOverlays(data);
  } catch (err) {
    console.error("Error loading case:", err);
  }
}

function renderCaseDetails(data) {
  const c = data.case;
  const rev = data.review;

  // Case Badge & Priority
  document.getElementById("caseBadge").innerText = `Case #${c.queue_index} of ${allCases.length}`;
  const pTag = document.getElementById("priorityTag");
  pTag.innerText = c.priority_reason.replace(/_/g, " ");
  pTag.className = `priority-tag priority-${Math.min(c.priority, 3)}`;

  // Metadata
  const sidEl = document.getElementById("studyUidShort");
  sidEl.innerText = `${c.study_id.slice(0, 15)}...${c.study_id.slice(-6)}`;
  sidEl.title = c.study_id;

  document.getElementById("catheterId").innerText = `Catheter #${c.catheter_instance_id} (of ${c.total_cvc_in_study} in study)`;
  document.getElementById("splitVal").innerText = c.split;
  document.getElementById("labelVal").innerText = c.device_label;
  document.getElementById("pointsVal").innerText = `${c.num_points} polyline points`;

  // Heuristic Details
  const hBox = document.getElementById("heuristicExplanation");
  const bBox = document.getElementById("borderDistDetails");
  hBox.innerHTML = `Heuristic (${c.heuristic_choice}) chooses endpoint with largest border distance.<br><strong>Candidate A:</strong> ${c.bdist_a}px | <strong>Candidate B:</strong> ${c.bdist_b}px (Diff: ${c.diff_bdist}px)`;
  bBox.innerText = `Anatomy: Distal CVC tip terminates centrally (SVC/CAJ). Entry is peripheral (neck/chest/groin).`;

  // Current Decision Selection
  resetButtonSelection();
  const notesEl = document.getElementById("reviewNotes");
  if (rev) {
    const dec = rev.reviewed_endpoint_class;
    if (dec === "A") btnChoiceA.classList.add("selected");
    else if (dec === "B") btnChoiceB.classList.add("selected");
    else btnChoiceU.classList.add("selected");
    notesEl.value = rev.audit_notes || "";
  } else {
    notesEl.value = "";
  }

  // Nav buttons
  btnPrev.disabled = (currentIndexInFiltered <= 0);
  btnNext.disabled = (currentIndexInFiltered >= currentFilteredList.length - 1);
}

function resetButtonSelection() {
  btnChoiceA.classList.remove("selected");
  btnChoiceB.classList.remove("selected");
  btnChoiceU.classList.remove("selected");
}

function renderRadiographAndOverlays(data) {
  const c = data.case;
  const sid = c.study_id;

  // Set image source
  radiographImg.onload = () => {
    setupSvgOverlay(data);
    fitToViewport();
  };
  radiographImg.src = `/api/image/${sid}`;
}

function setupSvgOverlay(data) {
  const c = data.case;
  const w = c.image_width;
  const h = c.image_height;

  svgOverlay.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svgOverlay.style.width = `${w}px`;
  svgOverlay.style.height = `${h}px`;
  radiographImg.style.width = `${w}px`;
  radiographImg.style.height = `${h}px`;

  // Clear SVG
  svgOverlay.innerHTML = "";

  // 1. Render other catheters in study (context)
  for (const other of data.all_study_catheters) {
    if (!other.is_current && other.pts && other.pts.length > 1) {
      const pathStr = other.pts.map(p => `${p[0]},${p[1]}`).join(" ");
      const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      poly.setAttribute("points", pathStr);
      poly.setAttribute("fill", "none");
      poly.setAttribute("stroke", "#64748b");
      poly.setAttribute("stroke-width", "5");
      poly.setAttribute("stroke-dasharray", "8,8");
      poly.setAttribute("opacity", "0.6");
      svgOverlay.appendChild(poly);
    }
  }

  // 2. Render Active Catheter Polyline
  if (c.polyline && c.polyline.length > 1) {
    const pathStr = c.polyline.map(p => `${p[0]},${p[1]}`).join(" ");
    
    // Shadow / Halo line
    const halo = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    halo.setAttribute("points", pathStr);
    halo.setAttribute("fill", "none");
    halo.setAttribute("stroke", "#0284c7");
    halo.setAttribute("stroke-width", "12");
    halo.setAttribute("opacity", "0.4");
    halo.setAttribute("stroke-linecap", "round");
    halo.setAttribute("stroke-linejoin", "round");
    svgOverlay.appendChild(halo);

    // Sharp foreground line
    const mainLine = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    mainLine.setAttribute("points", pathStr);
    mainLine.setAttribute("fill", "none");
    mainLine.setAttribute("stroke", "#38bdf8");
    mainLine.setAttribute("stroke-width", "6");
    mainLine.setAttribute("stroke-linecap", "round");
    mainLine.setAttribute("stroke-linejoin", "round");
    svgOverlay.appendChild(mainLine);
  }

  // 3. Render Endpoint A (Green)
  const ax = c.endpoint_a[0];
  const ay = c.endpoint_a[1];
  createEndpointMarker(ax, ay, "A", "#10b981");

  // 4. Render Endpoint B (Magenta)
  const bx = c.endpoint_b[0];
  const by = c.endpoint_b[1];
  createEndpointMarker(bx, by, "B", "#ec4899");
}

function createEndpointMarker(x, y, label, color) {
  const g = document.createElementNS("http://www.w3.org/2000/svg", "g");

  // Pulsing outer halo
  const pulse = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  pulse.setAttribute("cx", x);
  pulse.setAttribute("cy", y);
  pulse.setAttribute("r", "20");
  pulse.setAttribute("fill", color);
  pulse.setAttribute("fill-opacity", "0.25");
  pulse.setAttribute("stroke", color);
  pulse.setAttribute("stroke-width", "2");
  pulse.setAttribute("class", "pulse-circle");
  g.appendChild(pulse);

  // Center solid point
  const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  dot.setAttribute("cx", x);
  dot.setAttribute("cy", y);
  dot.setAttribute("r", "8");
  dot.setAttribute("fill", color);
  dot.setAttribute("stroke", "#ffffff");
  dot.setAttribute("stroke-width", "2");
  g.appendChild(dot);

  // Label badge
  const bgRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bgRect.setAttribute("x", x + 12);
  bgRect.setAttribute("y", y - 14);
  bgRect.setAttribute("width", "26");
  bgRect.setAttribute("height", "26");
  bgRect.setAttribute("rx", "6");
  bgRect.setAttribute("fill", color);
  bgRect.setAttribute("stroke", "#ffffff");
  bgRect.setAttribute("stroke-width", "2");
  g.appendChild(bgRect);

  const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
  text.setAttribute("x", x + 25);
  text.setAttribute("y", y + 4);
  text.setAttribute("text-anchor", "middle");
  text.setAttribute("fill", "#ffffff");
  text.setAttribute("font-size", "15");
  text.setAttribute("font-weight", "800");
  text.setAttribute("font-family", "system-ui, sans-serif");
  text.textContent = label;
  g.appendChild(text);

  svgOverlay.appendChild(g);
}

// Pan & Zoom Mechanics
function fitToViewport() {
  if (!radiographImg.naturalWidth) return;
  const vw = viewport.clientWidth;
  const vh = viewport.clientHeight;
  const iw = radiographImg.naturalWidth;
  const ih = radiographImg.naturalHeight;

  scale = Math.min((vw - 40) / iw, (vh - 40) / ih);
  panX = (vw - iw * scale) / 2;
  panY = (vh - ih * scale) / 2;
  updateTransform();
}

function updateTransform() {
  viewportContent.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
}

function updateImageFilters() {
  const contrast = contrastSlider.value;
  const brightness = brightnessSlider.value;
  const invert = isInverted ? "invert(100%)" : "";
  radiographImg.style.filter = `contrast(${contrast}%) brightness(${brightness}%) ${invert}`;
}

// Decision Action
async function submitDecision(decision) {
  if (!currentCaseData) return;
  const c = currentCaseData.case;
  const notes = document.getElementById("reviewNotes").value;

  resetButtonSelection();
  if (decision === "A") btnChoiceA.classList.add("selected");
  else if (decision === "B") btnChoiceB.classList.add("selected");
  else btnChoiceU.classList.add("selected");

  try {
    const res = await fetch("/api/decision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        study_id: c.study_id,
        catheter_instance_id: c.catheter_instance_id,
        decision: decision,
        notes: notes,
        reviewer_id: "interactive_expert_endpoint_review"
      })
    });
    const result = await res.json();
    if (result.success) {
      // Update local item
      const item = allCases.find(x => x.instance_key === c.instance_key);
      if (item) {
        item.is_reviewed = true;
        item.decision = decision;
      }
      refreshQueue();

      // Automatically advance to next case after short delay (300ms)
      setTimeout(() => {
        goNext();
      }, 300);
    }
  } catch (err) {
    console.error("Failed to submit decision:", err);
  }
}

function goNext() {
  if (currentIndexInFiltered < currentFilteredList.length - 1) {
    currentIndexInFiltered++;
    loadCaseByIndex(currentFilteredList[currentIndexInFiltered].queue_index);
  }
}

function goPrev() {
  if (currentIndexInFiltered > 0) {
    currentIndexInFiltered--;
    loadCaseByIndex(currentFilteredList[currentIndexInFiltered].queue_index);
  }
}

// Event Listeners
function setupEventListeners() {
  // Buttons
  btnChoiceA.addEventListener("click", () => submitDecision("A"));
  btnChoiceB.addEventListener("click", () => submitDecision("B"));
  btnChoiceU.addEventListener("click", () => submitDecision("unassessable"));

  btnNext.addEventListener("click", goNext);
  btnPrev.addEventListener("click", goPrev);

  filterPriority.addEventListener("change", () => {
    applyFilter();
    if (currentFilteredList.length > 0) {
      loadCaseByIndex(currentFilteredList[0].queue_index);
    }
  });

  // Toolbar
  document.getElementById("btnFit").addEventListener("click", fitToViewport);
  document.getElementById("btnReset").addEventListener("click", () => {
    contrastSlider.value = 100;
    brightnessSlider.value = 100;
    isInverted = false;
    btnInvert.classList.remove("active");
    updateImageFilters();
    fitToViewport();
  });
  document.getElementById("btnZoomIn").addEventListener("click", () => {
    scale *= 1.25;
    updateTransform();
  });
  document.getElementById("btnZoomOut").addEventListener("click", () => {
    scale /= 1.25;
    updateTransform();
  });
  btnInvert.addEventListener("click", () => {
    isInverted = !isInverted;
    btnInvert.classList.toggle("active", isInverted);
    updateImageFilters();
  });

  contrastSlider.addEventListener("input", updateImageFilters);
  brightnessSlider.addEventListener("input", updateImageFilters);

  // Pan & Zoom Mouse Drag
  viewport.addEventListener("mousedown", (e) => {
    isDragging = true;
    startX = e.clientX - panX;
    startY = e.clientY - panY;
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    panX = e.clientX - startX;
    panY = e.clientY - startY;
    updateTransform();
  });

  window.addEventListener("mouseup", () => {
    isDragging = false;
  });

  // Wheel Zoom centered at cursor
  viewport.addEventListener("wheel", (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    const rect = viewport.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    panX = mouseX - (mouseX - panX) * zoomFactor;
    panY = mouseY - (mouseY - panY) * zoomFactor;
    scale *= zoomFactor;
    updateTransform();
  }, { passive: false });

  // Keyboard Shortcuts
  window.addEventListener("keydown", (e) => {
    // If typing in notes input, don't trigger shortcuts
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    const key = e.key.toLowerCase();
    if (key === "a" || key === "1") {
      submitDecision("A");
    } else if (key === "b" || key === "2") {
      submitDecision("B");
    } else if (key === "u" || key === "3") {
      submitDecision("unassessable");
    } else if (key === "arrowright") {
      goNext();
    } else if (key === "arrowleft") {
      goPrev();
    } else if (key === "i") {
      isInverted = !isInverted;
      btnInvert.classList.toggle("active", isInverted);
      updateImageFilters();
    } else if (key === "f") {
      fitToViewport();
    }
  });

  window.addEventListener("resize", fitToViewport);
}

// Start
window.addEventListener("DOMContentLoaded", init);
