/**
 * script.js
 * ---------
 * Sign Language Translator — Frontend Logic
 *
 * Flow:
 *   1. Request camera + microphone permissions via getUserMedia.
 *   2. Mirror the stream to <video>.
 *   3. On "Start Translation": open WebSocket to backend, begin
 *      capturing frames at TARGET_FPS and sending as base64 JPEGs.
 *   4. Parse JSON responses { sign, text, confidence, status } and
 *      update the UI: sign chip, confidence bar, text output.
 *   5. Handle connection states (connecting / connected / disconnected /
 *      reconnecting) with visible indicators.
 */

'use strict';

// ═══════════════════════════════════════════════════════════════════
//  CONFIGURATION
// ═══════════════════════════════════════════════════════════════════

const CONFIG = {
  wsUrl:        `ws://${location.hostname}:8000/ws/translate`,
  targetFps:    15,     // increased for faster recognition response
  jpegQuality:  0.6,    // slightly lower quality = smaller payload = faster
  captureWidth:  320,   // smaller frame = much faster MediaPipe processing
  captureHeight: 240,
  reconnectDelay:  2000,  // ms before first reconnect attempt
  maxReconnects:   5,     // give up after N attempts
};

// ═══════════════════════════════════════════════════════════════════
//  DOM REFERENCES
// ═══════════════════════════════════════════════════════════════════

const $ = id => document.getElementById(id);

const videoEl         = $('video');
const captureCanvas   = $('capture-canvas');
const landmarkCanvas  = $('landmark-canvas');
const cameraOverlay   = $('camera-overlay');
const cameraSection   = $('camera-section');
const recordingBadge  = $('recording-badge');
const recLabel        = $('rec-label');
const fpsValue        = $('fps-value');
const connectionPill  = $('connection-pill');
const connLabel       = $('conn-label');
const currentSignEl   = $('current-sign');
const confidenceBar   = $('confidence-bar');
const confidenceValue = $('confidence-value');
const translationOut  = $('translation-output');
const outputTip       = $('output-tip');
const btnStartStop    = $('btn-start-stop');
const btnPermission   = $('btn-permission');
const btnClear        = $('btn-clear');
const btnCopy         = $('btn-copy');
// Camera-pane sign overlay
const cameraSignChip  = $('camera-sign-chip');
const cameraConfBar   = $('camera-conf-bar');
const cameraConfValue = $('camera-conf-value');

// ═══════════════════════════════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════════════════════════════

const state = {
  stream:         null,   // MediaStream
  micTrack:       null,   // AudioTrack (kept for future use)
  ws:             null,   // WebSocket instance
  capturing:      false,
  captureTimer:   null,
  reconnectCount: 0,
  reconnectTimer: null,
  framesSent:     0,
  fpsInterval:    null,
  lastSign:       '',
};

// Set up capture canvas (hidden)
captureCanvas.width  = CONFIG.captureWidth;
captureCanvas.height = CONFIG.captureHeight;
const captureCtx = captureCanvas.getContext('2d');

// ═══════════════════════════════════════════════════════════════════
//  TOAST NOTIFICATIONS
// ═══════════════════════════════════════════════════════════════════

let toastContainer = null;

function showToast(message, type = 'info', duration = 3000) {
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.className = 'toast-container';
    document.body.appendChild(toastContainer);
  }
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(40px)';
    setTimeout(() => toast.remove(), 350);
  }, duration);
}

// ═══════════════════════════════════════════════════════════════════
//  CAMERA PERMISSION
// ═══════════════════════════════════════════════════════════════════

async function requestCameraAccess() {
  try {
    // Request both video AND audio (mic track kept for future use)
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: true,
    });

    state.stream   = stream;
    state.micTrack = stream.getAudioTracks()[0] || null;

    videoEl.srcObject = stream;
    await videoEl.play();

    cameraOverlay.classList.add('hidden');
    showToast('📷 Camera ready!', 'success');
    return true;

  } catch (err) {
    console.error('[camera] getUserMedia failed:', err);
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      showToast('Camera permission denied. Please allow access.', 'error', 5000);
    } else if (err.name === 'NotFoundError') {
      showToast('No camera found on this device.', 'error', 5000);
    } else {
      showToast(`Camera error: ${err.message}`, 'error', 5000);
    }
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════
//  WEBSOCKET
// ═══════════════════════════════════════════════════════════════════

function setConnectionState(s) {
  // s: 'connecting' | 'connected' | 'disconnected' | 'reconnecting'
  connectionPill.className = `status-pill ${s}`;
  const labels = {
    connecting:   '⏳ Connecting…',
    connected:    'Connected',
    disconnected: 'Disconnected',
    reconnecting: '🔄 Reconnecting…',
  };
  connLabel.textContent = labels[s] || s;
}

function openWebSocket() {
  if (state.ws && state.ws.readyState <= 1) return; // already open/connecting

  setConnectionState('connecting');
  const ws = new WebSocket(CONFIG.wsUrl);
  state.ws = ws;

  ws.addEventListener('open', () => {
    state.reconnectCount = 0;
    setConnectionState('connected');
    showToast('🔗 Backend connected!', 'success');
  });

  ws.addEventListener('message', event => {
    try { handleServerMessage(JSON.parse(event.data)); }
    catch (e) { console.warn('[WS] Bad message:', event.data); }
  });

  ws.addEventListener('close', event => {
    setConnectionState('disconnected');
    if (state.capturing) attemptReconnect();
  });

  ws.addEventListener('error', () => {/* close follows */});
}

function attemptReconnect() {
  if (state.reconnectCount >= CONFIG.maxReconnects) {
    showToast('❌ Could not reconnect to backend.', 'error', 5000);
    stopCapture();
    return;
  }
  state.reconnectCount++;
  const delay = CONFIG.reconnectDelay * state.reconnectCount;
  setConnectionState('reconnecting');
  state.reconnectTimer = setTimeout(openWebSocket, delay);
}

function closeWebSocket() {
  if (state.ws) {
    state.ws.onclose = null;
    state.ws.close(1000, 'User stopped');
    state.ws = null;
  }
  setConnectionState('disconnected');
}

// ═══════════════════════════════════════════════════════════════════
//  SERVER MESSAGE HANDLER
// ═══════════════════════════════════════════════════════════════════

function handleServerMessage(msg) {
  const { sign = '', text = '', confidence = 0, status = '' } = msg;

  // ── Sign chip (right panel + camera overlay) ──
  const noHand = (sign === 'No hand' || sign === 'ERROR' || status === 'no_hand' || !sign);

  if (!noHand) {
    currentSignEl.textContent = sign;
    cameraSignChip.textContent = sign;
    if (sign !== state.lastSign) {
      state.lastSign = sign;
      currentSignEl.classList.remove('confirmed');
      void currentSignEl.offsetWidth; // restart animation
      currentSignEl.classList.add('confirmed');
    }
  } else {
    // No hand in frame — clear chips immediately
    currentSignEl.textContent = '—';
    cameraSignChip.textContent = '—';
    currentSignEl.classList.remove('confirmed');
    state.lastSign = '';
  }

  // ── Confidence bar (both panels) ──
  const pct = noHand ? 0 : Math.round(confidence * 100);
  confidenceBar.style.width   = `${pct}%`;
  confidenceValue.textContent = `${pct}%`;
  cameraConfBar.style.width   = `${pct}%`;
  cameraConfValue.textContent = `${pct}%`;

  // ── Translated text ──
  if (text && text !== translationOut.value) {
    translationOut.value = text;
    translationOut.scrollTop = translationOut.scrollHeight;
    outputTip.classList.add('hidden');
  }
}

// ═══════════════════════════════════════════════════════════════════
//  FRAME CAPTURE LOOP
// ═══════════════════════════════════════════════════════════════════

function captureFrame() {
  if (!state.capturing) return;
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  if (videoEl.readyState < 2) return;

  captureCtx.drawImage(videoEl, 0, 0, CONFIG.captureWidth, CONFIG.captureHeight);
  const dataUrl = captureCanvas.toDataURL('image/jpeg', CONFIG.jpegQuality);
  state.ws.send(dataUrl);
  state.framesSent++;
}

function startCapture() {
  if (state.capturing) return;
  state.capturing = true;

  state.captureTimer = setInterval(captureFrame, 1000 / CONFIG.targetFps);

  // FPS display
  let lastCount = 0;
  state.fpsInterval = setInterval(() => {
    fpsValue.textContent = state.framesSent - lastCount;
    lastCount = state.framesSent;
  }, 1000);

  cameraSection.classList.add('scanning');
  recordingBadge.classList.add('active');
  recLabel.textContent = 'LIVE';

  btnStartStop.classList.add('recording');
  btnStartStop.querySelector('.btn-icon').textContent  = '⏹';
  btnStartStop.querySelector('.btn-label').textContent = 'Stop Translation';
}

function stopCapture() {
  state.capturing = false;
  clearInterval(state.captureTimer);
  clearInterval(state.fpsInterval);
  clearTimeout(state.reconnectTimer);

  cameraSection.classList.remove('scanning');
  recordingBadge.classList.remove('active');
  recLabel.textContent = 'IDLE';
  fpsValue.textContent = '0';

  btnStartStop.classList.remove('recording');
  btnStartStop.querySelector('.btn-icon').textContent  = '▶';
  btnStartStop.querySelector('.btn-label').textContent = 'Start Translation';

  currentSignEl.textContent = '—';
  currentSignEl.classList.remove('confirmed');
  confidenceBar.style.width  = '0%';
  confidenceValue.textContent = '0%';

  closeWebSocket();
  showToast('Translation stopped.', 'info');
}

// ═══════════════════════════════════════════════════════════════════
//  LANDMARK CANVAS RESIZE
// ═══════════════════════════════════════════════════════════════════

function resizeLandmarkCanvas() {
  landmarkCanvas.width  = cameraSection.offsetWidth;
  landmarkCanvas.height = cameraSection.offsetHeight;
}
window.addEventListener('resize', resizeLandmarkCanvas);
resizeLandmarkCanvas();

// ═══════════════════════════════════════════════════════════════════
//  BUTTON HANDLERS
// ═══════════════════════════════════════════════════════════════════

// Grant camera (overlay button)
btnPermission.addEventListener('click', async () => {
  btnPermission.disabled = true;
  btnPermission.querySelector('.btn-label').textContent = '⏳ Requesting…';
  const ok = await requestCameraAccess();
  if (!ok) {
    btnPermission.disabled = false;
    btnPermission.querySelector('.btn-label').textContent = 'Grant Camera Access';
  }
});

// Start / Stop
btnStartStop.addEventListener('click', async () => {
  if (!state.stream) {
    const ok = await requestCameraAccess();
    if (!ok) return;
  }
  if (state.capturing) {
    stopCapture();
  } else {
    openWebSocket();
    startCapture();
    showToast('▶ Translation started!', 'success');
  }
});

// Clear
btnClear.addEventListener('click', () => {
  translationOut.value = '';
  outputTip.classList.remove('hidden');
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send('CLEAR');
  }
  showToast('🗑 Text cleared.', 'info');
});

// Copy
btnCopy.addEventListener('click', async () => {
  const text = translationOut.value.trim();
  if (!text) { showToast('Nothing to copy yet.', 'info'); return; }
  try {
    await navigator.clipboard.writeText(text);
    showToast('📋 Copied to clipboard!', 'success');
  } catch {
    showToast('Could not access clipboard.', 'error');
  }
});

// ═══════════════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════════════

(async function init() {
  console.log('[App] Sign Language Translator initialising…');
  setConnectionState('disconnected');
  // Try to acquire camera on page load for smoother UX
  await requestCameraAccess();
})();
