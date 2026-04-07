/**
 * Main Application Controller
 * Orchestrates WebSocket, Video, Charts, and DOM updates.
 */

(function () {
    'use strict';

    // ---- State ----
    const state = {
        pipelineRunning: false,
        equipment: {},  // tracker_id -> latest data
        totalFrames: 0
    };

    // ---- Module Instances ----
    const ws = new WSClient();
    const video = new VideoDisplay('video-canvas');
    const charts = new DashboardCharts();

    // ---- DOM References ----
    const dom = {
        connectionStatus: document.getElementById('connection-status'),
        statusText: document.querySelector('.status-text'),
        liveClock: document.getElementById('live-clock'),
        btnPipeline: document.getElementById('btn-start-pipeline'),
        totalEquipment: document.getElementById('total-equipment'),
        activeCount: document.getElementById('active-count'),
        inactiveCount: document.getElementById('inactive-count'),
        avgUtilization: document.getElementById('avg-utilization'),
        equipmentGrid: document.getElementById('equipment-grid'),
        equipmentCount: document.getElementById('equipment-count'),
        emptyState: document.getElementById('empty-state'),
        toastContainer: document.getElementById('toast-container'),
    };

    // ---- Clock ----
    function updateClock() {
        const now = new Date();
        dom.liveClock.textContent = now.toLocaleTimeString([], {
            hour: '2-digit', minute: '2-digit', second: '2-digit'
        });
    }
    setInterval(updateClock, 1000);
    updateClock();

    // ---- Toast Notifications ----
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        dom.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'toast-out 0.3s forwards';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // ---- Pipeline Control ----
    dom.btnPipeline.addEventListener('click', () => {
        if (!state.pipelineRunning) {
            if (ws.isConnected) {
                ws.sendCommand('start_pipeline');
                state.pipelineRunning = true;
                dom.btnPipeline.classList.add('running');
                dom.btnPipeline.querySelector('span').textContent = 'Stop Pipeline';
                dom.btnPipeline.querySelector('svg').innerHTML = '<rect x="4" y="4" width="8" height="8" fill="currentColor"/>';
                showToast('CV Pipeline started', 'success');
            } else {
                // Try REST API fallback
                fetch('/api/pipeline/start', { method: 'POST' })
                    .then(r => r.json())
                    .then(data => {
                        state.pipelineRunning = true;
                        dom.btnPipeline.classList.add('running');
                        dom.btnPipeline.querySelector('span').textContent = 'Stop Pipeline';
                        showToast('Pipeline started via API', 'success');
                    })
                    .catch(() => showToast('Failed to start pipeline', 'error'));
            }
        } else {
            if (ws.isConnected) {
                ws.sendCommand('stop_pipeline');
            } else {
                fetch('/api/pipeline/stop', { method: 'POST' }).catch(() => {});
            }
            state.pipelineRunning = false;
            dom.btnPipeline.classList.remove('running');
            dom.btnPipeline.querySelector('span').textContent = 'Start Pipeline';
            dom.btnPipeline.querySelector('svg').innerHTML = '<path d="M4 2L14 8L4 14V2Z" fill="currentColor"/>';
            showToast('Pipeline stopped', 'info');
        }
    });

    // ---- WebSocket Handlers ----

    ws.onConnect = () => {
        dom.connectionStatus.classList.add('connected');
        dom.statusText.textContent = 'Connected';
        showToast('Connected to server', 'success');
    };

    ws.onDisconnect = () => {
        dom.connectionStatus.classList.remove('connected');
        dom.statusText.textContent = 'Disconnected';
    };

    ws.onFrame = (base64Data) => {
        video.displayFrame(base64Data);
        state.totalFrames++;
    };

    ws.onEvent = (eventData) => {
        const id = eventData.equipment_id;
        state.equipment[id] = eventData;

        // Update equipment card
        updateEquipmentCard(id, eventData);

        // Update charts
        charts.updateActivity(eventData.activity);
        charts.addTimelinePoint(eventData.utilization || 0, eventData.motion_score || 0);

        // Update detection count
        video.setDetectionCount(Object.keys(state.equipment).length);
    };

    ws.onSummary = (summaryData) => {
        updateSummaryStats(summaryData);
    };

    ws.onStatus = (statusData) => {
        showToast(statusData.message || 'Status update', 'info');
    };

    // ---- Summary Stats ----
    function updateSummaryStats(summary) {
        const total = summary.total_equipment || 0;
        const active = summary.active_count || 0;
        const inactive = summary.inactive_count || 0;
        const avgUtil = summary.avg_utilization || 0;

        animateNumber(dom.totalEquipment, total);
        animateNumber(dom.activeCount, active);
        animateNumber(dom.inactiveCount, inactive);
        dom.avgUtilization.textContent = Math.round(avgUtil * 100) + '%';

        dom.equipmentCount.textContent = `${total} tracked`;
        charts.updateGauge(avgUtil);
    }

    function animateNumber(el, target) {
        const current = parseInt(el.textContent) || 0;
        if (current === target) return;
        el.textContent = target;
        el.style.transform = 'scale(1.15)';
        el.style.transition = 'transform 0.2s';
        setTimeout(() => {
            el.style.transform = 'scale(1)';
        }, 200);
    }

    // ---- Equipment Cards ----
    function updateEquipmentCard(id, data) {
        let card = document.getElementById(`eq-card-${id}`);

        if (!card) {
            // Create new card
            card = createEquipmentCard(id, data);
            dom.equipmentGrid.appendChild(card);

            // Remove empty state
            if (dom.emptyState) {
                dom.emptyState.style.display = 'none';
            }
        }

        // Update card content
        const isActive = data.state === 'ACTIVE';
        card.className = `card equipment-card ${isActive ? 'active' : ''}`;

        const badge = card.querySelector('.eq-badge');
        badge.className = `badge ${isActive ? 'badge-active' : 'badge-inactive'}`;
        badge.textContent = data.state;

        card.querySelector('.eq-activity').textContent = data.activity || 'Waiting';

        const activeTime = data.active_time || 0;
        const idleTime = data.idle_time || 0;
        card.querySelector('.eq-active-time').textContent = formatTime(activeTime);
        card.querySelector('.eq-idle-time').textContent = formatTime(idleTime);

        const utilPct = Math.round((data.utilization || 0) * 100);
        card.querySelector('.eq-util-value').textContent = utilPct + '%';
        card.querySelector('.utilization-bar-fill').style.width = utilPct + '%';

        // Algorithm badge
        const algoBadge = card.querySelector('.eq-algorithm');
        if (algoBadge) {
            algoBadge.textContent = (data.algorithm || 'rules').toUpperCase();
        }
    }

    function createEquipmentCard(id, data) {
        const card = document.createElement('div');
        card.id = `eq-card-${id}`;
        card.className = 'card equipment-card';

        card.innerHTML = `
            <div class="eq-header">
                <span class="eq-id">Equipment <span>#${id}</span></span>
                <span class="badge badge-inactive eq-badge">${data.state || 'INACTIVE'}</span>
            </div>
            <div class="eq-body">
                <div class="eq-metric">
                    <span class="eq-metric-label">Activity</span>
                    <span class="eq-metric-value eq-activity">${data.activity || 'Waiting'}</span>
                </div>
                <div class="eq-metric">
                    <span class="eq-metric-label">Algorithm</span>
                    <span class="eq-metric-value eq-algorithm" style="font-size:0.7rem;opacity:0.7;">${(data.algorithm || 'rules').toUpperCase()}</span>
                </div>
                <div class="eq-metric">
                    <span class="eq-metric-label">Active Time</span>
                    <span class="eq-metric-value eq-active-time">${formatTime(data.active_time || 0)}</span>
                </div>
                <div class="eq-metric">
                    <span class="eq-metric-label">Idle Time</span>
                    <span class="eq-metric-value eq-idle-time">${formatTime(data.idle_time || 0)}</span>
                </div>
                <div class="eq-utilization-bar">
                    <div class="utilization-bar-track">
                        <div class="utilization-bar-fill" style="width: 0%"></div>
                    </div>
                    <div class="utilization-bar-label">
                        <span>Utilization</span>
                        <span class="eq-util-value">0%</span>
                    </div>
                </div>
            </div>
        `;

        return card;
    }

    // ---- Utilities ----
    function formatTime(seconds) {
        if (seconds < 60) return Math.round(seconds) + 's';
        if (seconds < 3600) return Math.floor(seconds / 60) + 'm ' + Math.round(seconds % 60) + 's';
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        return h + 'h ' + m + 'm';
    }

    // ---- Polling Fallback (when WS is not available) ----
    async function pollMetrics() {
        if (ws.isConnected) return; // Skip if WS is working

        try {
            const res = await fetch('/api/metrics');
            if (res.ok) {
                const data = await res.json();
                updateSummaryStats(data);

                if (data.equipment) {
                    for (const eq of data.equipment) {
                        updateEquipmentCard(eq.equipment_id, eq);
                    }
                }
            }
        } catch (e) {
            // Silent fail for polling
        }
    }

    // Poll every 5 seconds as fallback
    setInterval(pollMetrics, 5000);

    // ---- Video Upload ----
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('video-file-input');
    const btnUpload = document.getElementById('btn-upload');
    const uploadProgress = document.getElementById('upload-progress');
    const uploadProgressText = document.getElementById('upload-progress-text');

    // Click to browse
    btnUpload.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    uploadZone.addEventListener('click', (e) => {
        if (e.target === uploadZone || e.target.closest('.overlay-content')) {
            fileInput.click();
        }
    });

    // File selected via browse
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadVideoFile(e.target.files[0]);
        }
    });

    // Drag & drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadZone.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            uploadVideoFile(files[0]);
        }
    });

    // Prevent default drag on the whole document
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());

    function uploadVideoFile(file) {
        // Validate
        const allowedTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm', 'video/x-flv', 'video/x-ms-wmv'];
        const allowedExts = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'];
        const ext = '.' + file.name.split('.').pop().toLowerCase();

        if (!allowedExts.includes(ext)) {
            showToast(`Unsupported format: ${ext}. Use MP4, AVI, MOV, MKV`, 'error');
            return;
        }

        const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
        showToast(`Uploading: ${file.name} (${sizeMB} MB)`, 'info');

        // Show progress
        uploadProgress.style.display = 'block';
        uploadProgressText.textContent = `Uploading ${file.name}...`;
        btnUpload.style.display = 'none';

        // Upload via XHR for progress tracking
        const formData = new FormData();
        formData.append('file', file);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/api/upload', true);

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const pct = Math.round((e.loaded / e.total) * 100);
                uploadProgressText.textContent = `Uploading... ${pct}%`;
            }
        };

        xhr.onload = () => {
            uploadProgress.style.display = 'none';
            btnUpload.style.display = 'inline-flex';

            if (xhr.status === 200) {
                const res = JSON.parse(xhr.responseText);
                if (res.status === 'success') {
                    showToast(`✓ ${res.filename} uploaded — Pipeline starting`, 'success');

                    // Update pipeline button state
                    state.pipelineRunning = true;
                    dom.btnPipeline.classList.add('running');
                    dom.btnPipeline.querySelector('span').textContent = 'Stop Pipeline';
                    dom.btnPipeline.querySelector('svg').innerHTML = '<rect x="4" y="4" width="8" height="8" fill="currentColor"/>';
                } else {
                    showToast(`Upload failed: ${res.message}`, 'error');
                }
            } else {
                showToast('Upload failed — server error', 'error');
            }
        };

        xhr.onerror = () => {
            uploadProgress.style.display = 'none';
            btnUpload.style.display = 'inline-flex';
            showToast('Upload failed — network error', 'error');
        };

        xhr.send(formData);
    }

    // ---- YouTube URL Download ----
    const urlInput = document.getElementById('video-url-input');
    const btnUrlDownload = document.getElementById('btn-url-download');

    btnUrlDownload.addEventListener('click', () => {
        const url = urlInput.value.trim();
        if (!url) {
            showToast('Please paste a video URL', 'error');
            urlInput.focus();
            return;
        }
        downloadFromUrl(url);
    });

    // Enter key to submit URL
    urlInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            btnUrlDownload.click();
        }
    });

    // Prevent upload zone click when interacting with URL input
    urlInput.addEventListener('click', (e) => e.stopPropagation());
    btnUrlDownload.addEventListener('click', (e) => e.stopPropagation());

    async function downloadFromUrl(url) {
        showToast('Downloading video... this may take a moment', 'info');

        // Show progress
        uploadProgress.style.display = 'block';
        uploadProgressText.textContent = 'Downloading from URL...';
        btnUrlDownload.disabled = true;
        btnUpload.style.display = 'none';

        try {
            const response = await fetch('/api/download-url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: url })
            });

            const data = await response.json();

            uploadProgress.style.display = 'none';
            btnUrlDownload.disabled = false;
            btnUpload.style.display = 'inline-flex';

            if (data.status === 'success') {
                const title = data.title || 'Video';
                const sizeMB = data.size_mb || 0;
                showToast(`✓ "${title}" downloaded (${sizeMB} MB) — Pipeline starting`, 'success');

                // Update pipeline button state
                state.pipelineRunning = true;
                dom.btnPipeline.classList.add('running');
                dom.btnPipeline.querySelector('span').textContent = 'Stop Pipeline';
                dom.btnPipeline.querySelector('svg').innerHTML = '<rect x="4" y="4" width="8" height="8" fill="currentColor"/>';

                // Clear URL input
                urlInput.value = '';
            } else {
                showToast(`Download failed: ${data.message}`, 'error');
            }
        } catch (err) {
            uploadProgress.style.display = 'none';
            btnUrlDownload.disabled = false;
            btnUpload.style.display = 'inline-flex';
            showToast('Download failed — network error', 'error');
        }
    }

    // ---- Initialize ----
    ws.connect();
    console.log('[App] Equipment Monitor dashboard initialized');

})();
