// CVC Distal-Tip Visual Annotation Tool - Client Application
// Implements native zoom/pan, polyline overlays, click-to-mark tips (no auto-snapping), and study completion attestation.

const CATHETER_PALETTE = [
  '#38bdf8', // Sky
  '#f43f5e', // Rose
  '#10b981', // Emerald
  '#a855f7', // Purple
  '#f59e0b', // Amber
  '#3b82f6', // Blue
  '#ec4899', // Pink
];

class AnnotationApp {
  constructor() {
    this.currentQueue = 'provisional_pilot_25';
    this.studies = [];
    this.currentStudyIdx = -1;
    this.currentStudyData = null;
    this.activeCatheterId = null;
    this.markMode = 'tip';

    // Viewport transform
    this.scale = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isDragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;
    this.mouseDownPos = { x: 0, y: 0 };
    this.invertContrast = false;
    this.showPolylines = true;

    // Current study review working state
    this.catheterReviews = {}; // { [cid]: { state: string, marked_distal_tip: {x, y}|null, notes: string } }
    this.carinaReview = { state: null, coordinate: null, notes: '' };
    this.uncatalogedCatheters = []; // [ { marked_distal_tip: {x, y}, state: 'assessable', notes: string } ]

    // Elements
    this.canvas = document.getElementById('main-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.img = new Image();
    this.imgLoaded = false;

    this.initDOM();
    this.bindEvents();
    this.loadQueues();
  }

  initDOM() {
    this.queueSelect = document.getElementById('queue-select');
    this.reviewerInput = document.getElementById('reviewer-input');
    this.studyListEl = document.getElementById('study-list');
    this.studySearch = document.getElementById('study-search');
    this.spinner = document.getElementById('canvas-loading-spinner');
    this.cursorCoordsBadge = document.getElementById('cursor-native-coords');
    this.zoomLevelBadge = document.getElementById('zoom-level-badge');
    this.studyIndexIndicator = document.getElementById('study-index-indicator');

    // Restore saved reviewer name from localStorage
    const savedReviewer = localStorage.getItem('cvc_reviewer_id');
    if (savedReviewer) {
      this.reviewerInput.value = savedReviewer;
    }
  }

  bindEvents() {
    // Window resize
    window.addEventListener('resize', () => this.resizeCanvas());

    // Reviewer input save
    this.reviewerInput.addEventListener('change', () => {
      localStorage.setItem('cvc_reviewer_id', this.reviewerInput.value.trim());
      this.loadStudiesForQueue();
    });

    // Queue select
    this.queueSelect.addEventListener('change', (e) => {
      this.currentQueue = e.target.value;
      this.loadStudiesForQueue();
    });

    // Navigation buttons
    document.getElementById('btn-prev-study').addEventListener('click', () => this.prevStudy());
    document.getElementById('btn-next-study').addEventListener('click', () => this.nextStudy());
    document.getElementById('btn-save-review').addEventListener('click', () => this.saveReview(false));
    document.getElementById('btn-attest-complete').addEventListener('click', () => this.saveReview(true));
    document.getElementById('btn-mark-tip').addEventListener('click', () => this.setMarkMode('tip'));
    document.getElementById('btn-mark-carina').addEventListener('click', () => this.setMarkMode('carina'));
    document.getElementById('carina-state').addEventListener('change', (e) => {
      this.carinaReview.state = e.target.value || null;
      if (this.carinaReview.state !== 'visible') this.carinaReview.coordinate = null;
      this.updateCarinaDisplay(); this.updateStudyCompletionCheck(); this.render();
    });
    document.getElementById('carina-notes').addEventListener('input', (e) => { this.carinaReview.notes = e.target.value; });
    document.getElementById('btn-clear-carina').addEventListener('click', () => {
      this.carinaReview.coordinate = null; this.updateCarinaDisplay(); this.updateStudyCompletionCheck(); this.render();
    });
    document.getElementById('placement-label').addEventListener('change', (e) => {
      if (this.activeCatheterId === null) return;
      this.catheterReviews[this.activeCatheterId].placement_label = e.target.value || null;
      this.updateStudyCompletionCheck();
    });

    // Canvas zoom & pan
    const stage = document.getElementById('canvas-stage');
    stage.addEventListener('wheel', (e) => this.handleWheel(e), { passive: false });
    stage.addEventListener('mousedown', (e) => this.handleMouseDown(e));
    window.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    window.addEventListener('mouseup', (e) => this.handleMouseUp(e));

    // Toolbar buttons
    document.getElementById('btn-zoom-in').addEventListener('click', () => this.zoom(1.25));
    document.getElementById('btn-zoom-out').addEventListener('click', () => this.zoom(0.8));
    document.getElementById('btn-zoom-reset').addEventListener('click', () => this.resetZoom());
    document.getElementById('btn-zoom-100').addEventListener('click', () => this.zoom100());

    document.getElementById('toggle-invert').addEventListener('change', (e) => {
      this.invertContrast = e.target.checked;
      this.render();
    });
    document.getElementById('toggle-polylines').addEventListener('change', (e) => {
      this.showPolylines = e.target.checked;
      this.render();
    });

    // Catheter State Pill Buttons
    ['assessable', 'uncertain', 'outside_image', 'annotation_incorrect'].forEach((state) => {
      const btn = document.getElementById(`pill-${state === 'outside_image' ? 'outside' : state === 'annotation_incorrect' ? 'incorrect' : state}`);
      if (btn) {
        btn.addEventListener('click', () => this.setCatheterState(state));
      }
    });

    // Clear coordinate button
    document.getElementById('btn-clear-tip').addEventListener('click', () => this.clearActiveTipCoordinate());

    // Catheter notes input
    document.getElementById('cath-notes-input').addEventListener('input', (e) => {
      if (this.activeCatheterId !== null && this.catheterReviews[this.activeCatheterId]) {
        this.catheterReviews[this.activeCatheterId].notes = e.target.value;
      }
    });

    // Uncataloged CVC
    document.getElementById('btn-add-uncataloged').addEventListener('click', () => this.addUncatalogedCatheter());

    // Search filter
    this.studySearch.addEventListener('input', () => this.filterStudyList());
    document.querySelectorAll('.filter-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.filter-btn').forEach((b) => b.classList.remove('active'));
        e.target.classList.add('active');
        this.filterStudyList();
      });
    });

    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => this.handleKeyDown(e));

    // Image onload
    this.img.onload = () => {
      this.imgLoaded = true;
      this.spinner.classList.add('hidden');
      this.resetZoom();
      this.render();
    };
  }

  async loadQueues() {
    try {
      const res = await fetch('/api/queues');
      const queues = await res.json();
      this.queueSelect.innerHTML = queues
        .map((q) => `<option value="${q.id}">${q.name}</option>`)
        .join('');
      if (queues.length > 0) {
        this.currentQueue = queues[0].id;
        this.loadStudiesForQueue();
      }
    } catch (err) {
      console.error('Failed to load queues:', err);
    }
  }

  async loadStudiesForQueue() {
    try {
      const reviewer = encodeURIComponent(this.reviewerInput.value.trim());
      const res = await fetch(`/api/studies?queue_name=${this.currentQueue}&reviewer_id=${reviewer}`);
      const data = await res.json();
      this.studies = data.studies;
      this.renderStudyList();
      if (this.studies.length > 0) {
        this.selectStudy(0);
      }
    } catch (err) {
      console.error('Failed to load studies:', err);
    }
  }

  renderStudyList() {
    this.studyListEl.innerHTML = this.studies
      .map((s, idx) => {
        const isCompleted = s.review_state === 'completed';
        const badgeClass = isCompleted ? 'completed' : 'pending';
        const badgeText = isCompleted ? 'DONE' : 'PENDING';
        const uidShort = s.study_id.slice(0, 16) + '...';
        return `
          <li class="study-item ${idx === this.currentStudyIdx ? 'active' : ''}" data-idx="${idx}" onclick="app.selectStudy(${idx})">
            <div class="study-item-title">
              <span>${uidShort}</span>
              <span class="status-badge ${badgeClass}">${badgeText}</span>
            </div>
            <div class="study-item-meta">
              <span>Pt: ${s.patient_id}</span>
              <span>•</span>
              <span>${s.cvc_count} CVCs</span>
            </div>
          </li>
        `;
      })
      .join('');
  }

  filterStudyList() {
    const query = this.studySearch.value.toLowerCase().trim();
    const activeFilter = document.querySelector('.filter-btn.active').dataset.filter;

    const items = document.querySelectorAll('.study-item');
    items.forEach((item) => {
      const idx = parseInt(item.dataset.idx, 10);
      const study = this.studies[idx];
      const matchQuery = study.study_id.toLowerCase().includes(query) || study.patient_id.toLowerCase().includes(query);
      const matchStatus =
        activeFilter === 'all' ||
        (activeFilter === 'completed' && study.review_state === 'completed') ||
        (activeFilter === 'pending' && study.review_state !== 'completed');

      item.style.display = matchQuery && matchStatus ? 'block' : 'none';
    });
  }

  async selectStudy(idx) {
    if (idx < 0 || idx >= this.studies.length) return;
    this.currentStudyIdx = idx;
    this.updateStudyIndicator();

    const study = this.studies[idx];
    this.spinner.classList.remove('hidden');
    this.imgLoaded = false;

    // Highlight active in list
    document.querySelectorAll('.study-item').forEach((el, i) => {
      el.classList.toggle('active', i === idx);
    });

    try {
      const reviewer = encodeURIComponent(this.reviewerInput.value.trim());
      const res = await fetch(`/api/study/${study.study_id}?queue_name=${this.currentQueue}&reviewer_id=${reviewer}`);
      const data = await res.json();
      this.currentStudyData = data;

      // Populate study metadata
      document.getElementById('current-study-uid').textContent = data.study_id;
      document.getElementById('current-patient-id').textContent = data.patient_id;
      document.getElementById('current-image-dims').textContent = `${data.image_width} × ${data.image_height} px`;
      document.getElementById('catheter-count-badge').textContent = data.catheters.length;

      // Initialize or restore review state
      this.initStudyReviewState(data);

      // Render catheter cards
      this.renderCatheterCards();

      // Load native image
      this.img.src = `/api/image/${study.study_id}`;

      // Select first catheter by default
      if (data.catheters.length > 0) {
        this.selectCatheter(data.catheters[0].catheter_instance_id);
      }
    } catch (err) {
      console.error('Failed to load study details:', err);
      this.spinner.classList.add('hidden');
    }
  }

  initStudyReviewState(data) {
    this.catheterReviews = {};
    this.uncatalogedCatheters = [];

    const existing = data.existing_review;
    this.carinaReview = existing && existing.carina_review
      ? { state: null, coordinate: null, notes: '', ...existing.carina_review }
      : { state: null, coordinate: null, notes: '' };
    document.getElementById('carina-state').value = this.carinaReview.state || '';
    document.getElementById('carina-notes').value = this.carinaReview.notes || '';
    this.updateCarinaDisplay();
    if (existing) {
      document.getElementById('study-notes-input').value = existing.review_notes || '';
      this.uncatalogedCatheters = existing.uncataloged_catheters || [];
      data.catheters.forEach((c) => {
        const cidStr = String(c.catheter_instance_id);
        if (existing.catheter_reviews && existing.catheter_reviews[cidStr]) {
          this.catheterReviews[c.catheter_instance_id] = { ...existing.catheter_reviews[cidStr] };
        } else {
          this.catheterReviews[c.catheter_instance_id] = {
            state: null,
            marked_distal_tip: null,
            placement_label: null,
            notes: '',
          };
        }
      });
    } else {
      document.getElementById('study-notes-input').value = '';
      data.catheters.forEach((c) => {
        // STRICT REQUIREMENT: No automatically accepted endpoint suggestions.
        this.catheterReviews[c.catheter_instance_id] = {
          state: null,
          marked_distal_tip: null,
          placement_label: null,
          notes: '',
        };
      });
    }

    this.renderUncatalogedList();
    this.updateStudyCompletionCheck();
  }

  renderCatheterCards() {
    const container = document.getElementById('catheter-cards-container');
    container.innerHTML = this.currentStudyData.catheters
      .map((c, i) => {
        const color = CATHETER_PALETTE[i % CATHETER_PALETTE.length];
        const rev = this.catheterReviews[c.catheter_instance_id];
        const state = rev ? rev.state : null;
        const stateClass = state || 'unreviewed';
        const stateText = state ? state.replace('_', ' ').toUpperCase() : 'UNREVIEWED';
        const isActive = c.catheter_instance_id === this.activeCatheterId;

        return `
          <div class="cath-card ${isActive ? 'active' : ''}" data-cid="${c.catheter_instance_id}" onclick="app.selectCatheter(${c.catheter_instance_id})">
            <div class="cath-card-left">
              <div class="cath-color-indicator" style="background-color: ${color}"></div>
              <div>
                <span class="cath-id-text">CVC #${c.catheter_instance_id}</span>
                <span class="cath-label-pill">${c.label}</span>
              </div>
            </div>
            <span class="cath-state-badge ${stateClass}">${stateText}</span>
          </div>
        `;
      })
      .join('');
  }

  selectCatheter(cid) {
    this.activeCatheterId = cid;
    this.renderCatheterCards();

    const cath = this.currentStudyData.catheters.find((c) => c.catheter_instance_id === cid);
    if (!cath) return;

    const idx = this.currentStudyData.catheters.indexOf(cath);
    const color = CATHETER_PALETTE[idx % CATHETER_PALETTE.length];

    document.getElementById('active-cath-color-dot').style.backgroundColor = color;
    document.getElementById('active-cath-title').textContent = `Reviewing CVC #${cid} (${cath.label})`;

    // Update state pill buttons
    const rev = this.catheterReviews[cid] || { state: null, marked_distal_tip: null, notes: '' };
    document.querySelectorAll('.pill-btn').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.state === rev.state);
    });

    // Update marked coordinate display
    const coordEl = document.getElementById('marked-tip-coords');
    if (rev.marked_distal_tip) {
      coordEl.textContent = `X: ${rev.marked_distal_tip.x.toFixed(1)} px, Y: ${rev.marked_distal_tip.y.toFixed(1)} px`;
    } else {
      coordEl.textContent = 'No coordinate marked (Click image to mark)';
    }

    // Update notes
    document.getElementById('cath-notes-input').value = rev.notes || '';
    document.getElementById('placement-label').value = rev.placement_label || '';

    this.render();
  }

  setCatheterState(state) {
    if (this.activeCatheterId === null) return;
    const rev = this.catheterReviews[this.activeCatheterId];
    rev.state = state;

    if (state !== 'assessable') {
      // If uncertain/outside/incorrect, coordinate is not mandatory
    }

    this.selectCatheter(this.activeCatheterId);
    this.updateStudyCompletionCheck();
  }

  clearActiveTipCoordinate() {
    if (this.activeCatheterId === null) return;
    this.catheterReviews[this.activeCatheterId].marked_distal_tip = null;
    this.selectCatheter(this.activeCatheterId);
    this.updateStudyCompletionCheck();
  }

  updateStudyCompletionCheck() {
    if (!this.currentStudyData) return;
    const catheters = this.currentStudyData.catheters;
    let allReviewed = true;
    let missingReason = '';

    for (const c of catheters) {
      const rev = this.catheterReviews[c.catheter_instance_id];
      if (!rev || !rev.state) {
        allReviewed = false;
        missingReason = `CVC #${c.catheter_instance_id} has no assigned state.`;
        break;
      }
      if (rev.state === 'assessable' && !rev.marked_distal_tip) {
        allReviewed = false;
        missingReason = `CVC #${c.catheter_instance_id} is marked 'Assessable' but lacks tip coordinate.`;
        break;
      }
      if (this.currentQueue === 'carina_placement_pilot_50' && !rev.placement_label) {
        allReviewed = false; missingReason = `CVC #${c.catheter_instance_id} needs a placement label.`; break;
      }
    }
    if (allReviewed && this.currentQueue === 'carina_placement_pilot_50') {
      if (!this.carinaReview || !this.carinaReview.state) {
        allReviewed = false; missingReason = 'Assign a carina state (visible, uncertain, or not visible).';
      } else if (this.carinaReview.state === 'visible' && !this.carinaReview.coordinate) {
        allReviewed = false; missingReason = 'Carina is visible; mark its coordinate on the image.';
      }
    }

    const iconEl = document.getElementById('study-complete-icon');
    const textEl = document.getElementById('study-complete-text');
    const rowEl = document.querySelector('.status-indicator-row');

    if (allReviewed) {
      rowEl.classList.add('ready');
      iconEl.textContent = '✅';
      textEl.textContent = 'Required landmarks and CVC labels reviewed. Ready to attest.';
    } else {
      rowEl.classList.remove('ready');
      iconEl.textContent = '⚠️';
      textEl.textContent = missingReason;
    }

    return allReviewed;
  }

  addUncatalogedCatheter() {
    const uncatIdx = this.uncatalogedCatheters.length + 1;
    const item = {
      marked_distal_tip: { x: this.currentStudyData.image_width / 2, y: this.currentStudyData.image_height / 2 },
      state: 'assessable',
      notes: `Uncataloged CVC #${uncatIdx}`,
    };
    this.uncatalogedCatheters.push(item);
    this.renderUncatalogedList();
    this.render();
  }

  renderUncatalogedList() {
    const listEl = document.getElementById('uncataloged-list');
    const hintEl = document.getElementById('uncataloged-empty-hint');
    if (this.uncatalogedCatheters.length === 0) {
      hintEl.style.display = 'block';
      listEl.innerHTML = '';
      return;
    }
    hintEl.style.display = 'none';
    listEl.innerHTML = this.uncatalogedCatheters
      .map(
        (u, idx) => `
        <li style="font-size: 0.8rem; padding: 0.35rem 0; border-bottom: 1px dashed rgba(255,255,255,0.06);">
          <span>Extra CVC #${idx + 1}: (${u.marked_distal_tip.x.toFixed(0)}, ${u.marked_distal_tip.y.toFixed(0)})</span>
          <button class="btn-xs btn-outline" onclick="app.removeUncataloged(${idx})">Remove</button>
        </li>
      `
      )
      .join('');
  }

  removeUncataloged(idx) {
    this.uncatalogedCatheters.splice(idx, 1);
    this.renderUncatalogedList();
    this.render();
  }

  async saveReview(attestComplete = false) {
    if (!this.currentStudyData) return;
    const reviewerId = this.reviewerInput.value.trim() || 'engineering_provisional';
    const isComplete = this.updateStudyCompletionCheck();

    if (attestComplete && !isComplete) {
      alert('Cannot attest completion: Please review all catheters and mark coordinates for assessable tips first.');
      return;
    }

    const payload = {
      study_id: this.currentStudyData.study_id,
      queue_name: this.currentQueue,
      reviewer_id: reviewerId,
      review_notes: document.getElementById('study-notes-input').value.trim(),
      catheter_reviews: this.catheterReviews,
      carina_review: this.carinaReview,
      uncataloged_catheters: this.uncatalogedCatheters,
      is_study_complete: isComplete,
    };

    try {
      const res = await fetch('/api/save_review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Review could not be saved');
      if (data.status === 'success') {
        this.studies[this.currentStudyIdx].review_state = isComplete ? 'completed' : 'in_progress';
        this.renderStudyList();

        if (attestComplete) {
          this.nextStudy();
        }
      }
    } catch (err) {
      console.error('Save failed:', err);
      alert('Failed to save review: ' + err.message);
    }
  }

  nextStudy() {
    if (this.currentStudyIdx < this.studies.length - 1) {
      this.selectStudy(this.currentStudyIdx + 1);
    }
  }

  prevStudy() {
    if (this.currentStudyIdx > 0) {
      this.selectStudy(this.currentStudyIdx - 1);
    }
  }

  updateStudyIndicator() {
    this.studyIndexIndicator.textContent = `${this.currentStudyIdx + 1} / ${this.studies.length}`;
  }

  // ---------------- Canvas Viewport & Drawing ----------------
  resizeCanvas() {
    const stage = document.getElementById('canvas-stage');
    this.canvas.width = stage.clientWidth;
    this.canvas.height = stage.clientHeight;
    this.render();
  }

  resetZoom() {
    if (!this.imgLoaded) return;
    const stage = document.getElementById('canvas-stage');
    this.canvas.width = stage.clientWidth;
    this.canvas.height = stage.clientHeight;

    const scaleX = this.canvas.width / this.img.width;
    const scaleY = this.canvas.height / this.img.height;
    this.scale = Math.min(scaleX, scaleY) * 0.95;
    this.panX = (this.canvas.width - this.img.width * this.scale) / 2;
    this.panY = (this.canvas.height - this.img.height * this.scale) / 2;
    this.render();
  }

  zoom100() {
    if (!this.imgLoaded) return;
    this.scale = 1.0;
    this.panX = (this.canvas.width - this.img.width) / 2;
    this.panY = (this.canvas.height - this.img.height) / 2;
    this.render();
  }

  zoom(factor) {
    const cx = this.canvas.width / 2;
    const cy = this.canvas.height / 2;
    this.scale *= factor;
    this.scale = Math.max(0.1, Math.min(this.scale, 10.0));
    this.panX = cx - (cx - this.panX) * factor;
    this.panY = cy - (cy - this.panY) * factor;
    this.render();
  }

  handleWheel(e) {
    e.preventDefault();
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    const newScale = Math.max(0.1, Math.min(this.scale * factor, 15.0));
    const effectiveFactor = newScale / this.scale;

    this.panX = mouseX - (mouseX - this.panX) * effectiveFactor;
    this.panY = mouseY - (mouseY - this.panY) * effectiveFactor;
    this.scale = newScale;

    this.render();
  }

  handleMouseDown(e) {
    if (e.button !== 0 && e.button !== 1) return;
    this.isDragging = true;
    this.dragStartX = e.clientX - this.panX;
    this.dragStartY = e.clientY - this.panY;
    this.mouseDownPos = { x: e.clientX, y: e.clientY };
  }

  handleMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (this.isDragging) {
      this.panX = e.clientX - this.dragStartX;
      this.panY = e.clientY - this.dragStartY;
      this.render();
    }

    if (this.imgLoaded) {
      const natX = (mouseX - this.panX) / this.scale;
      const natY = (mouseY - this.panY) / this.scale;
      if (natX >= 0 && natX <= this.img.width && natY >= 0 && natY <= this.img.height) {
        this.cursorCoordsBadge.textContent = `X: ${natX.toFixed(0)}, Y: ${natY.toFixed(0)}`;
      } else {
        this.cursorCoordsBadge.textContent = 'X: --- , Y: ---';
      }
    }
  }

  handleMouseUp(e) {
    if (!this.isDragging) return;
    this.isDragging = false;

    // Check if this was a click rather than a pan drag (< 4 px motion)
    const distMoved = Math.hypot(e.clientX - this.mouseDownPos.x, e.clientY - this.mouseDownPos.y);
    if (distMoved < 4 && this.imgLoaded) {
      const rect = this.canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const natX = (mouseX - this.panX) / this.scale;
      const natY = (mouseY - this.panY) / this.scale;

      if (natX >= 0 && natX <= this.img.width && natY >= 0 && natY <= this.img.height) {
        this.onImageClicked(natX, natY);
      }
    }
  }

  onImageClicked(natX, natY) {
    if (this.markMode === 'carina') {
      this.carinaReview.state = 'visible';
      this.carinaReview.coordinate = { x: parseFloat(natX.toFixed(1)), y: parseFloat(natY.toFixed(1)) };
      document.getElementById('carina-state').value = 'visible';
      this.updateCarinaDisplay(); this.updateStudyCompletionCheck(); this.render();
      return;
    }
    if (this.activeCatheterId === null) return;
    const rev = this.catheterReviews[this.activeCatheterId];
    if (!rev) return;

    // Set marked distal tip coordinate explicitly (NO AUTO SNAPPING)
    rev.marked_distal_tip = {
      x: parseFloat(natX.toFixed(1)),
      y: parseFloat(natY.toFixed(1)),
    };
    rev.state = 'assessable';

    this.selectCatheter(this.activeCatheterId);
    this.updateStudyCompletionCheck();
    this.render();
  }

  render() {
    if (!this.imgLoaded) return;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    ctx.save();
    ctx.translate(this.panX, this.panY);
    ctx.scale(this.scale, this.scale);

    // Apply contrast inversion if toggled
    if (this.invertContrast) {
      ctx.filter = 'invert(1)';
    } else {
      ctx.filter = 'none';
    }

    // Draw radiograph
    ctx.drawImage(this.img, 0, 0);
    ctx.filter = 'none';

    // Draw CVC polylines
    if (this.showPolylines && this.currentStudyData) {
      this.currentStudyData.catheters.forEach((c, i) => {
        const color = CATHETER_PALETTE[i % CATHETER_PALETTE.length];
        const isActive = c.catheter_instance_id === this.activeCatheterId;
        this.drawPolyline(c.polyline, color, isActive);
      });
    }

    // Draw marked distal tips for all reviewed catheters
    if (this.currentStudyData) {
      this.currentStudyData.catheters.forEach((c, i) => {
        const rev = this.catheterReviews[c.catheter_instance_id];
        if (rev && rev.marked_distal_tip) {
          const color = CATHETER_PALETTE[i % CATHETER_PALETTE.length];
          const isActive = c.catheter_instance_id === this.activeCatheterId;
          this.drawMarkedTip(rev.marked_distal_tip.x, rev.marked_distal_tip.y, color, isActive, `CVC #${c.catheter_instance_id}`);
        }
      });

      // Draw uncataloged tips
      this.uncatalogedCatheters.forEach((u, i) => {
        this.drawMarkedTip(u.marked_distal_tip.x, u.marked_distal_tip.y, '#f59e0b', false, `Extra #${i + 1}`);
      });
      const p = this.carinaReview && this.carinaReview.coordinate;
      if (p) this.drawMarkedTip(p.x, p.y, '#fde047', false, 'Carina');
    }

    ctx.restore();

    this.zoomLevelBadge.textContent = `${Math.round(this.scale * 100)}%`;
  }

  setMarkMode(mode) {
    this.markMode = mode;
    document.getElementById('btn-mark-tip').classList.toggle('active', mode === 'tip');
    document.getElementById('btn-mark-carina').classList.toggle('active', mode === 'carina');
    document.getElementById('canvas-guidance-text').textContent = mode === 'carina'
      ? '💡 Click the carina on the radiograph. Drag to pan; scroll to zoom. No auto-snapping.'
      : '💡 Click to mark the selected CVC tip. Drag to pan; scroll to zoom. No auto-snapping.';
  }

  updateCarinaDisplay() {
    const p = this.carinaReview && this.carinaReview.coordinate;
    document.getElementById('carina-coords').textContent = p
      ? `X: ${p.x.toFixed(1)} px, Y: ${p.y.toFixed(1)} px`
      : 'No coordinate marked';
  }

  drawPolyline(pts, color, isActive) {
    if (!pts || pts.length < 2) return;
    const ctx = this.ctx;
    ctx.save();

    ctx.strokeStyle = color;
    ctx.lineWidth = (isActive ? 4.0 : 2.5) / this.scale;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    if (isActive) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
    }

    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i][0], pts[i][1]);
    }
    ctx.stroke();

    ctx.restore();
  }

  drawMarkedTip(x, y, color, isActive, label) {
    const ctx = this.ctx;
    ctx.save();

    const radius = (isActive ? 7.0 : 5.0) / this.scale;
    const crosshair = 14.0 / this.scale;

    // Glowing circle
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = isActive ? 16 : 8;

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    // Crosshair
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5 / this.scale;
    ctx.beginPath();
    ctx.moveTo(x - crosshair, y);
    ctx.lineTo(x + crosshair, y);
    ctx.moveTo(x, y - crosshair);
    ctx.lineTo(x, y + crosshair);
    ctx.stroke();

    // Text label
    ctx.font = `${Math.max(12, 14 / this.scale)}px 'JetBrains Mono', monospace`;
    ctx.fillStyle = '#ffffff';
    ctx.shadowColor = '#000000';
    ctx.shadowBlur = 4;
    ctx.fillText(label, x + crosshair + 4 / this.scale, y - 4 / this.scale);

    ctx.restore();
  }

  handleKeyDown(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    if (e.key === '[' || e.key === 'ArrowLeft') {
      this.prevStudy();
    } else if (e.key === ']' || e.key === 'ArrowRight') {
      this.nextStudy();
    } else if (e.key.toLowerCase() === 'r') {
      this.resetZoom();
    } else if (e.key.toLowerCase() === 'a') {
      this.setCatheterState('assessable');
    } else if (e.key.toLowerCase() === 'u') {
      this.setCatheterState('uncertain');
    } else if (e.key.toLowerCase() === 'o') {
      this.setCatheterState('outside_image');
    } else if (e.key.toLowerCase() === 'i') {
      this.setCatheterState('annotation_incorrect');
    } else if (e.key.toLowerCase() === 'c') {
      this.clearActiveTipCoordinate();
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
      e.preventDefault();
      this.saveReview(false);
    } else if (e.key >= '1' && e.key <= '9') {
      const idx = parseInt(e.key, 10) - 1;
      if (this.currentStudyData && idx < this.currentStudyData.catheters.length) {
        this.selectCatheter(this.currentStudyData.catheters[idx].catheter_instance_id);
      }
    }
  }
}

// Instantiate on load
window.app = null;
window.addEventListener('DOMContentLoaded', () => {
  window.app = new AnnotationApp();
});
