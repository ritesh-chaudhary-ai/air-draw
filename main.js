/* ============================================================
   AIR DRAW — Main Application
   Gesture-based interactive doodler powered by MediaPipe
   ============================================================ */

// ── State ──────────────────────────────────────────────────────
const state = {
  handLandmarker: null,
  webcamStream: null,
  isReady: false,
  // Drawing
  strokes: [],           // Array of { points: [{x,y}], color, thickness, glow }
  currentStroke: null,
  activeColor: '#00f0ff',
  thickness: 6,
  glowIntensity: 60,
  // Gesture
  currentGesture: 'idle',
  previousGesture: 'idle',
  gestureStableFrames: 0,
  gestureStartTime: 0,
  isModalOpen: true,
  // Grab & Move
  isGrabbing: false,
  grabStartPos: null,
  grabOffset: { x: 0, y: 0 },
  totalOffset: { x: 0, y: 0 },
  nearestStrokeIdx: -1,
  pinchMode: null, // 'grab' | 'colorpicker'
  // Gesture color picker (radial hue wheel)
  colorPicker: {
    active: false,
    center: { x: 0, y: 0 },
    angle: 0,
    radius: 90,
    previewColor: null,
  },
  // Erase
  eraserRadius: 28,
  // Camera
  showCamera: true,
  cameraOpacity: 0.35,
  // Particles
  particles: [],
  // Smoothing
  smoothPos: { x: 0, y: 0 },
  smoothFactor: 0.35,
  // Canvas dimensions
  width: 0,
  height: 0,
  // Audio
  audioCtx: null,
};

// ── DOM Elements ───────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const loadingScreen = $('loading-screen');
const appEl = $('app');
const webcamEl = $('webcam');
const cameraCanvas = $('camera-canvas');
const drawingCanvas = $('drawing-canvas');
const uiCanvas = $('ui-canvas');
const cameraCtx = cameraCanvas.getContext('2d');
const drawingCtx = drawingCanvas.getContext('2d');
const uiCtx = uiCanvas.getContext('2d');
const gestureHud = $('gesture-hud');
const gestureIcon = $('gesture-icon');
const gestureLabel = $('gesture-label');
const thicknessSlider = $('thickness-slider');
const thicknessValue = $('thickness-value');
const glowSlider = $('glow-slider');
const glowValue = $('glow-value');

const cameraModeText = $('camera-mode-text');
const cameraModeIndicator = $('camera-mode-indicator');
const onboardingModal = $('onboarding-modal');
const btnStart = $('btn-start');

// ── Audio (subtle sound effects) ───────────────────────────────
function getAudioCtx() {
  if (!state.audioCtx) {
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  return state.audioCtx;
}

function playTone(freq, duration, type = 'sine', volume = 0.06) {
  try {
    const ctx = getAudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, ctx.currentTime);
    gain.gain.setValueAtTime(volume, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + duration);
  } catch (e) { /* audio not available */ }
}

function playDrawStart() { playTone(880, 0.08, 'sine', 0.04); }
function playDrawEnd() { playTone(440, 0.1, 'sine', 0.03); }
function playEraseSound() { playTone(200, 0.06, 'triangle', 0.03); }
function playGrabSound() { playTone(660, 0.1, 'sine', 0.05); }
function playDropSound() { playTone(330, 0.15, 'sine', 0.04); }
function playTrashSound() { 
  playTone(400, 0.1, 'sawtooth', 0.04); 
  setTimeout(() => playTone(200, 0.2, 'sawtooth', 0.05), 100);
}
function playModeSwitch() { playTone(1200, 0.05, 'sine', 0.03); }
function playColorPickOpen() { playTone(700, 0.06, 'sine', 0.04); }
function playColorPickConfirm() { playTone(1000, 0.08, 'sine', 0.05); }
function playSnapSound() {
  playTone(1400, 0.05, 'sine', 0.05);
  setTimeout(() => playTone(1900, 0.09, 'sine', 0.04), 55);
}

// ── Canvas Setup ───────────────────────────────────────────────
function resizeCanvases() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  state.width = w;
  state.height = h;
  [cameraCanvas, drawingCanvas, uiCanvas].forEach(c => {
    c.width = w;
    c.height = h;
  });
}

window.addEventListener('resize', () => {
  resizeCanvases();
  redrawStrokes();
});

// Phones fire 'orientationchange' before the viewport has actually
// resized — a short delay ensures innerWidth/innerHeight are correct.
window.addEventListener('orientationchange', () => {
  setTimeout(() => {
    resizeCanvases();
    redrawStrokes();
  }, 300);
});

const isMobileDevice = /Android|iPhone|iPad|iPod|Mobi/i.test(navigator.userAgent);

// ── MediaPipe Loading ──────────────────────────────────────────
async function initMediaPipe() {
  // Dynamic import from CDN
  const { FilesetResolver, HandLandmarker } = await import(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs'
  );

  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
  );

  const baseOptions = {
    modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    delegate: 'GPU'
  };

  try {
    state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions,
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence: 0.6,
      minTrackingConfidence: 0.5,
    });
  } catch (gpuError) {
    // Some mobile browsers (older Android WebViews, some iOS Safari
    // versions) don't support the WebGL GPU delegate — fall back to CPU.
    console.warn('GPU delegate failed, falling back to CPU:', gpuError);
    state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { ...baseOptions, delegate: 'CPU' },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.6,
      minHandPresenceConfidence: 0.6,
      minTrackingConfidence: 0.5,
    });
  }

  return true;
}

// ── Webcam Setup ───────────────────────────────────────────────
async function initWebcam() {
  // Ask for a lighter resolution on phones to keep frame rate smooth
  const idealWidth = isMobileDevice ? 640 : 1280;
  const idealHeight = isMobileDevice ? 480 : 720;

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: idealWidth }, height: { ideal: idealHeight }, facingMode: 'user' }
  });
  webcamEl.srcObject = stream;
  state.webcamStream = stream;

  return new Promise((resolve) => {
    webcamEl.onloadedmetadata = () => {
      webcamEl.play();
      resolve();
    };
  });
}


// ── Gesture Detection ──────────────────────────────────────────
function detectGesture(landmarks) {
  if (!landmarks || landmarks.length === 0) return 'none';

  const lm = landmarks;

  // Finger tip and pip/mcp landmarks
  const thumbTip = lm[4];
  const thumbIP = lm[3];
  const indexTip = lm[8];
  const indexPIP = lm[6];
  const indexMCP = lm[5];
  const middleTip = lm[12];
  const middlePIP = lm[10];
  const ringTip = lm[16];
  const ringPIP = lm[14];
  const pinkyTip = lm[20];
  const pinkyPIP = lm[18];

  // Finger extended checks (y decreases going up in normalized coords)
  const indexUp = indexTip.y < indexPIP.y - 0.02; // stricter up
  
  // Others must be strictly curled down (tip below PIP)
  const middleDown = middleTip.y > middlePIP.y;
  const ringDown = ringTip.y > ringPIP.y;
  const pinkyDown = pinkyTip.y > pinkyPIP.y;
  
  // Open palm check (original relaxed conditions)
  const middleUp = middleTip.y < middlePIP.y;
  const ringUp = ringTip.y < ringPIP.y;
  const pinkyUp = pinkyTip.y < pinkyPIP.y;
  const thumbOut = Math.abs(thumbTip.x - thumbIP.x) > 0.03 || thumbTip.y < thumbIP.y;

  // Pinch detection: thumb tip close to index tip
  const pinchDist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
  const isPinching = pinchDist < 0.06;

  // Gesture classification
  if (isPinching && !middleUp && !ringUp && !pinkyUp) {
    return 'pinch';
  }

  if (indexUp && middleUp && ringUp && pinkyUp && thumbOut) {
    return 'open_palm';
  }

  // Stricter drawing check: only index up, all others strictly down
  if (indexUp && middleDown && ringDown && pinkyDown) {
    return 'index_finger';
  }

  if (!indexUp && !middleUp && !ringUp && !pinkyUp) {
    return 'fist';
  }

  return 'idle';
}

// Stabilize gesture — require N consistent frames to switch
function stabilizeGesture(rawGesture) {
  // If same as current, stay put
  if (rawGesture === state.currentGesture) {
    state.previousGesture = rawGesture;
    state.gestureStableFrames = 0;
    return state.currentGesture;
  }

  // If this is the same candidate as last frame, count up
  if (rawGesture === state.previousGesture) {
    state.gestureStableFrames++;
  } else {
    // New candidate — reset counter
    state.previousGesture = rawGesture;
    state.gestureStableFrames = 1;
  }

  const threshold = rawGesture === 'pinch' ? 3 : 4;

  if (state.gestureStableFrames >= threshold) {
    const oldGesture = state.currentGesture;
    state.currentGesture = rawGesture;
    state.gestureStableFrames = 0;
    
    // Record start time of new gesture
    state.gestureStartTime = Date.now();

    if (oldGesture !== rawGesture) {
      onGestureChange(oldGesture, rawGesture);
    }
    return rawGesture;
  }

  return state.currentGesture;
}

function onGestureChange(from, to) {
  // Sound effects
  if (to === 'index_finger') playDrawStart();
  else if (to === 'open_palm') playModeSwitch();
  else if (to === 'pinch') playGrabSound();
  else if (from === 'index_finger') playDrawEnd();

  // End current stroke if we were drawing
  if (from === 'index_finger' && state.currentStroke) {
    finalizeStroke(state.currentStroke);
    state.currentStroke = null;
    redrawStrokes();
  }

  // End grab
  if (from === 'pinch') {
    endGrab();
  }

  // Update HUD
  updateGestureHUD(to);
}

function updateGestureHUD(gesture) {
  const map = {
    'index_finger': { icon: '☝️', label: 'Drawing', cls: 'drawing' },
    'open_palm':    { icon: '✋', label: 'Erasing', cls: 'erasing' },
    'pinch':        { icon: '🤏', label: 'Grab', cls: 'grabbing' },
    'fist':         { icon: '✊', label: 'Idle', cls: '' },
    'idle':         { icon: '🖐️', label: 'Ready', cls: '' },
    'none':         { icon: '👋', label: 'Show hand', cls: '' },
  };
  const info = map[gesture] || map['idle'];
  gestureIcon.textContent = info.icon;
  gestureLabel.textContent = info.label;
  gestureHud.className = info.cls;
}

// ── Shape Recognition ("Magic Snap") ────────────────────────────
// Analyzes a freshly-drawn stroke and, if it closely matches a
// circle, rectangle, or straight line, returns a description of
// the "perfect" shape to snap it to.
function detectShape(points) {
  const n = points.length;
  if (n < 8) return null;

  const start = points[0];
  const end = points[n - 1];
  const closeDist = Math.hypot(end.x - start.x, end.y - start.y);

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  let sumX = 0, sumY = 0;
  for (const p of points) {
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
    sumX += p.x; sumY += p.y;
  }
  const w = maxX - minX;
  const h = maxY - minY;
  const cx = sumX / n;
  const cy = sumY / n;
  const diag = Math.hypot(w, h);

  if (diag < 45) return null; // too small to bother snapping

  const isClosed = closeDist < diag * 0.28;

  // Straight line: max perpendicular deviation from the start-end chord
  if (!isClosed) {
    const dx = end.x - start.x, dy = end.y - start.y;
    const len = Math.hypot(dx, dy);
    if (len > 70) {
      let maxDev = 0;
      for (const p of points) {
        const t = ((p.x - start.x) * dx + (p.y - start.y) * dy) / (len * len);
        const projX = start.x + t * dx, projY = start.y + t * dy;
        maxDev = Math.max(maxDev, Math.hypot(p.x - projX, p.y - projY));
      }
      if (maxDev < len * 0.07 + 5) {
        return { type: 'line', start, end };
      }
    }
    return null;
  }

  // Circle: low variance in radius from centroid
  const radii = points.map(p => Math.hypot(p.x - cx, p.y - cy));
  const avgR = radii.reduce((a, b) => a + b, 0) / n;
  const variance = radii.reduce((a, r) => a + (r - avgR) * (r - avgR), 0) / n;
  const stdDev = Math.sqrt(variance);
  if (avgR > 25 && stdDev / avgR < 0.22) {
    return { type: 'circle', cx, cy, r: avgR };
  }

  // Rectangle: shoelace area vs. bounding-box area ratio
  let area = 0;
  for (let i = 0; i < n; i++) {
    const p1 = points[i], p2 = points[(i + 1) % n];
    area += p1.x * p2.y - p2.x * p1.y;
  }
  area = Math.abs(area) / 2;
  const boxArea = w * h;
  if (boxArea > 0 && w > 45 && h > 45 && area / boxArea > 0.62) {
    return { type: 'rectangle', minX, minY, w, h };
  }

  return null;
}

function generateShapePoints(shape) {
  if (shape.type === 'circle') {
    const pts = [];
    const steps = 56;
    for (let i = 0; i <= steps; i++) {
      const a = (i / steps) * Math.PI * 2;
      pts.push({ x: shape.cx + Math.cos(a) * shape.r, y: shape.cy + Math.sin(a) * shape.r });
    }
    return pts;
  }
  if (shape.type === 'rectangle') {
    const { minX, minY, w, h } = shape;
    return [
      { x: minX, y: minY },
      { x: minX + w, y: minY },
      { x: minX + w, y: minY + h },
      { x: minX, y: minY + h },
      { x: minX, y: minY },
    ];
  }
  // line
  return [shape.start, shape.end];
}

// Finalize a stroke: check for a shape match, snap + celebrate if found,
// otherwise push the freehand stroke as-is.
function finalizeStroke(stroke) {
  if (!stroke || stroke.points.length < 2) return;

  const shape = detectShape(stroke.points);
  if (shape) {
    const shapePoints = generateShapePoints(shape);
    const snapped = {
      points: shapePoints,
      color: stroke.color,
      thickness: stroke.thickness,
      glow: stroke.glow,
      straight: shape.type !== 'circle',
    };
    state.strokes.push(snapped);
    triggerSnapCelebration(shapePoints, stroke.color);
    playSnapSound();
    flashHudMessage('✨', 'Shape!');
  } else {
    state.strokes.push({ ...stroke });
  }
}

function triggerSnapCelebration(points, color) {
  const step = Math.max(1, Math.floor(points.length / 26));
  for (let i = 0; i < points.length; i += step) {
    const p = points[i];
    for (let j = 0; j < 2; j++) {
      state.particles.push({
        x: p.x, y: p.y,
        vx: (Math.random() - 0.5) * 4.5,
        vy: (Math.random() - 0.5) * 4.5,
        life: 1,
        decay: 0.012 + Math.random() * 0.02,
        size: 3 + Math.random() * 3.5,
        color,
      });
    }
  }
}

let hudFlashTimeout = null;
function flashHudMessage(icon, label) {
  gestureIcon.textContent = icon;
  gestureLabel.textContent = label;
  gestureHud.classList.add('shape-snap');
  clearTimeout(hudFlashTimeout);
  hudFlashTimeout = setTimeout(() => {
    gestureHud.classList.remove('shape-snap');
    updateGestureHUD(state.currentGesture);
  }, 700);
}

// ── Drawing Logic ──────────────────────────────────────────────
function getLandmarkPos(landmark) {
  // Mirror X for natural feel, and scale to canvas
  return {
    x: (1 - landmark.x) * state.width,
    y: landmark.y * state.height
  };
}

function smoothPosition(rawPos) {
  state.smoothPos.x += (rawPos.x - state.smoothPos.x) * state.smoothFactor;
  state.smoothPos.y += (rawPos.y - state.smoothPos.y) * state.smoothFactor;
  return { x: state.smoothPos.x, y: state.smoothPos.y };
}

function handleDrawing(landmarks) {
  const indexTip = landmarks[8];
  const rawPos = getLandmarkPos(indexTip);
  const pos = smoothPosition(rawPos);

  // Buffer: Ignore drawing for first 300ms to avoid trailing lines from transition
  if (Date.now() - state.gestureStartTime < 300) {
    state.smoothPos = { ...rawPos };
    return;
  }

  if (!state.currentStroke) {
    state.currentStroke = {
      points: [pos],
      color: state.activeColor,
      thickness: state.thickness,
      glow: state.glowIntensity,
    };
    state.smoothPos = { ...rawPos };
  } else {
    state.currentStroke.points.push({ ...pos });
  }

  // Emit particles
  emitParticles(pos.x, pos.y, state.activeColor);
  
  redrawStrokes();
}

function handleErasing(landmarks) {
  const wrist = landmarks[0];
  const middleMCP = landmarks[9];
  // Palm center is roughly between wrist and middle MCP
  const palmCenter = {
    x: (1 - (wrist.x + middleMCP.x) / 2) * state.width,
    y: ((wrist.y + middleMCP.y) / 2) * state.height
  };

  const radius = state.eraserRadius;
  let erased = false;

  // Segment-based erasing: split strokes, keeping only points outside eraser
  const newStrokes = [];
  for (let i = 0; i < state.strokes.length; i++) {
    const stroke = state.strokes[i];
    const segments = [];
    let currentSegment = [];

    for (const p of stroke.points) {
      const dx = p.x - palmCenter.x;
      const dy = p.y - palmCenter.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist >= radius) {
        // Point is outside eraser — keep it
        currentSegment.push(p);
      } else {
        // Point is inside eraser — break the stroke here
        erased = true;
        if (currentSegment.length >= 2) {
          segments.push(currentSegment);
        }
        currentSegment = [];
      }
    }

    // Don't forget the last segment
    if (currentSegment.length >= 2) {
      segments.push(currentSegment);
    }

    // Convert segments back to strokes
    if (segments.length === 0 && stroke.points.length > 0) {
      // Entire stroke was erased — count as erased
      // (don't add anything back)
    } else if (segments.length === 1 && segments[0].length === stroke.points.length) {
      // Stroke was untouched
      newStrokes.push(stroke);
    } else {
      // Stroke was split into pieces
      for (const seg of segments) {
        newStrokes.push({
          points: seg,
          color: stroke.color,
          thickness: stroke.thickness,
          glow: stroke.glow,
        });
      }
    }
  }

  state.strokes = newStrokes;

  if (erased) {
    playEraseSound();
  }

  // Draw eraser circle on UI canvas
  uiCtx.beginPath();
  uiCtx.arc(palmCenter.x, palmCenter.y, radius, 0, Math.PI * 2);
  uiCtx.strokeStyle = 'rgba(255, 45, 107, 0.5)';
  uiCtx.lineWidth = 1.5;
  uiCtx.setLineDash([5, 5]);
  uiCtx.stroke();
  uiCtx.setLineDash([]);
  uiCtx.fillStyle = 'rgba(255, 45, 107, 0.05)';
  uiCtx.fill();

  redrawStrokes();
}

function handleGrab(landmarks) {
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const pinchPos = {
    x: (1 - (thumbTip.x + indexTip.x) / 2) * state.width,
    y: ((thumbTip.y + indexTip.y) / 2) * state.height
  };

  if (!state.isGrabbing) {
    state.isGrabbing = true;
    state.grabStartPos = { ...pinchPos };

    const idx = findNearestStroke(pinchPos);
    if (idx >= 0) {
      // Pinching on top of an existing stroke → move it
      state.pinchMode = 'grab';
      state.nearestStrokeIdx = idx;
    } else {
      // Pinching in empty space → open the gesture color picker
      state.pinchMode = 'colorpicker';
      state.colorPicker.active = true;
      state.colorPicker.center = { ...pinchPos };
      state.colorPicker.previewColor = state.activeColor;
      playColorPickOpen();
    }
  } else if (state.pinchMode === 'grab') {
    const dx = pinchPos.x - state.grabStartPos.x;
    const dy = pinchPos.y - state.grabStartPos.y;

    if (state.nearestStrokeIdx >= 0 && state.nearestStrokeIdx < state.strokes.length) {
      const stroke = state.strokes[state.nearestStrokeIdx];
      const prevDx = state.grabOffset.x;
      const prevDy = state.grabOffset.y;
      const deltaDx = dx - prevDx;
      const deltaDy = dy - prevDy;

      for (let i = 0; i < stroke.points.length; i++) {
        stroke.points[i].x += deltaDx;
        stroke.points[i].y += deltaDy;
      }
    }

    state.grabOffset = { x: dx, y: dy };
  } else if (state.pinchMode === 'colorpicker') {
    const dx = pinchPos.x - state.colorPicker.center.x;
    const dy = pinchPos.y - state.colorPicker.center.y;
    const dist = Math.hypot(dx, dy);
    if (dist > 14) {
      state.colorPicker.angle = Math.atan2(dy, dx);
      const hue = ((state.colorPicker.angle * 180 / Math.PI) + 360) % 360;
      state.colorPicker.previewColor = hslToHex(hue, 88, 55);
    }
  }

  if (state.pinchMode === 'grab') {
    // Draw grab indicator
    uiCtx.beginPath();
    uiCtx.arc(pinchPos.x, pinchPos.y, 18, 0, Math.PI * 2);
    uiCtx.strokeStyle = 'rgba(255, 215, 0, 0.7)';
    uiCtx.lineWidth = 2;
    uiCtx.stroke();
    uiCtx.fillStyle = 'rgba(255, 215, 0, 0.1)';
    uiCtx.fill();

    // Highlight grabbed stroke
    if (state.nearestStrokeIdx >= 0 && state.nearestStrokeIdx < state.strokes.length) {
      drawStrokeHighlight(state.strokes[state.nearestStrokeIdx]);
    }
  } else if (state.pinchMode === 'colorpicker') {
    drawColorWheel(uiCtx);
  }

  redrawStrokes();
}

function endGrab() {
  if (state.pinchMode === 'grab' && state.isGrabbing && state.nearestStrokeIdx >= 0) {
    playDropSound();
  } else if (state.pinchMode === 'colorpicker' && state.colorPicker.active) {
    // Confirm the picked color
    state.activeColor = state.colorPicker.previewColor || state.activeColor;
    document.querySelectorAll('.color-swatch').forEach(b => b.classList.remove('active'));
    playColorPickConfirm();
    state.colorPicker.active = false;
  }

  state.isGrabbing = false;
  state.grabStartPos = null;
  state.grabOffset = { x: 0, y: 0 };
  state.nearestStrokeIdx = -1;
  state.pinchMode = null;
  redrawStrokes();
}

// ── Gesture Color Picker (radial hue wheel) ─────────────────────
function hslToHex(h, s, l) {
  s /= 100; l /= 100;
  const k = (n) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  const toHex = (x) => Math.round(255 * x).toString(16).padStart(2, '0');
  return `#${toHex(f(0))}${toHex(f(8))}${toHex(f(4))}`;
}

function drawColorWheel(ctx) {
  const { center, radius, angle, previewColor } = state.colorPicker;
  const segments = 60;

  ctx.save();

  // Hue ring
  for (let i = 0; i < segments; i++) {
    const a0 = (i / segments) * Math.PI * 2;
    const a1 = ((i + 1) / segments) * Math.PI * 2;
    const hue = (i / segments) * 360;
    ctx.beginPath();
    ctx.moveTo(center.x, center.y);
    ctx.arc(center.x, center.y, radius, a0, a1);
    ctx.closePath();
    ctx.fillStyle = `hsl(${hue}, 88%, 55%)`;
    ctx.fill();
  }

  // Punch a hole in the middle for a donut look
  ctx.globalCompositeOperation = 'destination-out';
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius * 0.42, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalCompositeOperation = 'source-over';

  // Selection pointer on the ring
  const px = center.x + Math.cos(angle) * radius * 0.71;
  const py = center.y + Math.sin(angle) * radius * 0.71;
  ctx.beginPath();
  ctx.arc(px, py, 9, 0, Math.PI * 2);
  ctx.fillStyle = '#fff';
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = previewColor || '#fff';
  ctx.stroke();

  // Center preview swatch
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius * 0.34, 0, Math.PI * 2);
  ctx.fillStyle = previewColor || '#fff';
  ctx.shadowColor = previewColor || '#fff';
  ctx.shadowBlur = 22;
  ctx.fill();

  ctx.restore();

  gestureIcon.textContent = '🎨';
  gestureLabel.textContent = 'Pick Color';
  gestureHud.className = 'picking';
}

function findNearestStroke(pos) {
  let minDist = Infinity;
  let nearestIdx = -1;

  for (let i = 0; i < state.strokes.length; i++) {
    const stroke = state.strokes[i];
    for (const p of stroke.points) {
      const d = Math.hypot(p.x - pos.x, p.y - pos.y);
      if (d < minDist) {
        minDist = d;
        nearestIdx = i;
      }
    }
  }

  return minDist < 80 ? nearestIdx : -1;
}



function drawStrokeHighlight(stroke) {
  if (!stroke || stroke.points.length < 2) return;
  uiCtx.save();
  uiCtx.beginPath();
  uiCtx.moveTo(stroke.points[0].x, stroke.points[0].y);
  for (let i = 1; i < stroke.points.length; i++) {
    uiCtx.lineTo(stroke.points[i].x, stroke.points[i].y);
  }
  uiCtx.strokeStyle = 'rgba(255, 215, 0, 0.3)';
  uiCtx.lineWidth = stroke.thickness + 12;
  uiCtx.lineCap = 'round';
  uiCtx.lineJoin = 'round';
  uiCtx.setLineDash([8, 8]);
  uiCtx.stroke();
  uiCtx.setLineDash([]);
  uiCtx.restore();
}

// ── Stroke Rendering with Glow ─────────────────────────────────
function traceStrokePath(ctx, pts, straight) {
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  if (straight) {
    // Sharp-cornered path — used for snapped rectangles / straight lines
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i].x, pts[i].y);
    }
  } else {
    for (let i = 1; i < pts.length; i++) {
      const prev = pts[i - 1];
      const curr = pts[i];
      const mx = (prev.x + curr.x) / 2;
      const my = (prev.y + curr.y) / 2;
      ctx.quadraticCurveTo(prev.x, prev.y, mx, my);
    }
    ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
  }
}

function drawGlowStroke(ctx, stroke, isCurrentStroke = false) {
  if (!stroke || stroke.points.length < 2) return;

  const pts = stroke.points;
  const color = stroke.color;
  const width = stroke.thickness;
  const glowMult = stroke.glow / 100;
  const straight = !!stroke.straight;

  ctx.save();
  ctx.lineCap = straight ? 'square' : 'round';
  ctx.lineJoin = straight ? 'miter' : 'round';

  // Pass 1: Outer glow
  if (glowMult > 0) {
    traceStrokePath(ctx, pts, straight);
    ctx.strokeStyle = color;
    ctx.lineWidth = width * 3;
    ctx.globalAlpha = 0.1 * glowMult;
    ctx.shadowColor = color;
    ctx.shadowBlur = 35 * glowMult;
    ctx.stroke();
  }

  // Pass 2: Mid glow
  if (glowMult > 0) {
    traceStrokePath(ctx, pts, straight);
    ctx.strokeStyle = color;
    ctx.lineWidth = width * 1.6;
    ctx.globalAlpha = 0.35 * glowMult;
    ctx.shadowBlur = 15 * glowMult;
    ctx.stroke();
  }

  // Pass 3: Core line
  traceStrokePath(ctx, pts, straight);
  ctx.strokeStyle = lightenColor(color, 0.5);
  ctx.lineWidth = width;
  ctx.globalAlpha = 1;
  ctx.shadowBlur = 6 * glowMult;
  ctx.shadowColor = color;
  ctx.stroke();

  ctx.restore();
}

function lightenColor(hex, amount) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  const nr = Math.min(255, Math.round(r + (255 - r) * amount));
  const ng = Math.min(255, Math.round(g + (255 - g) * amount));
  const nb = Math.min(255, Math.round(b + (255 - b) * amount));
  return `rgb(${nr}, ${ng}, ${nb})`;
}

function redrawStrokes() {
  drawingCtx.clearRect(0, 0, state.width, state.height);

  // Draw all completed strokes
  for (const stroke of state.strokes) {
    drawGlowStroke(drawingCtx, stroke);
  }

  // Draw current stroke
  if (state.currentStroke && state.currentStroke.points.length > 1) {
    drawGlowStroke(drawingCtx, state.currentStroke, true);
  }
}

// ── Particles ──────────────────────────────────────────────────
function emitParticles(x, y, color) {
  for (let i = 0; i < 3; i++) {
    state.particles.push({
      x, y,
      vx: (Math.random() - 0.5) * 2.2,
      vy: (Math.random() - 0.5) * 2.2,
      life: 1,
      decay: 0.018 + Math.random() * 0.026,
      size: 1.5 + Math.random() * 3.2,
      color,
    });
  }
}

function updateAndDrawParticles(ctx) {
  for (let i = state.particles.length - 1; i >= 0; i--) {
    const p = state.particles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.life -= p.decay;
    p.size *= 0.97;

    if (p.life <= 0) {
      state.particles.splice(i, 1);
      continue;
    }

    ctx.save();
    ctx.globalAlpha = p.life * 0.7;
    ctx.fillStyle = p.color;
    ctx.shadowColor = p.color;
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

// ── Hand Skeleton Drawing ──────────────────────────────────────
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],      // Thumb
  [0,5],[5,6],[6,7],[7,8],      // Index
  [0,9],[9,10],[10,11],[11,12], // Middle  (via 0→9)
  [0,13],[13,14],[14,15],[15,16], // Ring  (via 0→13)
  [0,17],[17,18],[18,19],[19,20], // Pinky (via 0→17)
  [5,9],[9,13],[13,17],          // Palm connections
];

function drawHandSkeleton(ctx, landmarks) {
  if (!landmarks) return;

  ctx.save();
  ctx.globalAlpha = 0.3;

  // Draw connections
  for (const [a, b] of HAND_CONNECTIONS) {
    const pa = getLandmarkPos(landmarks[a]);
    const pb = getLandmarkPos(landmarks[b]);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Draw landmarks
  for (let i = 0; i < landmarks.length; i++) {
    const pos = getLandmarkPos(landmarks[i]);
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.fill();
  }

  // Highlight fingertip
  const tips = [4, 8, 12, 16, 20];
  for (const t of tips) {
    const pos = getLandmarkPos(landmarks[t]);
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.shadowColor = '#ffffff';
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  ctx.restore();
}

// ── Drawing cursor indicator ───────────────────────────────────
function drawCursorIndicator(ctx, landmarks, gesture) {
  if (gesture === 'index_finger') {
    const pos = getLandmarkPos(landmarks[8]);
    // Outer ring
    ctx.save();
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, state.thickness / 2 + 6, 0, Math.PI * 2);
    ctx.strokeStyle = state.activeColor;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.5;
    ctx.shadowColor = state.activeColor;
    ctx.shadowBlur = 8;
    ctx.stroke();
    // Inner dot
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = state.activeColor;
    ctx.globalAlpha = 0.9;
    ctx.fill();
    ctx.restore();
  }
}

// ── Main Render Loop ───────────────────────────────────────────
let lastVideoTime = -1;

function renderLoop() {
  if (!state.handLandmarker || !state.isReady) {
    requestAnimationFrame(renderLoop);
    return;
  }

  const video = webcamEl;
  const now = performance.now();

  // Draw camera feed
  cameraCtx.clearRect(0, 0, state.width, state.height);
  if (state.showCamera) {
    cameraCtx.save();
    cameraCtx.globalAlpha = state.cameraOpacity;
    // Mirror the camera
    cameraCtx.translate(state.width, 0);
    cameraCtx.scale(-1, 1);
    cameraCtx.drawImage(video, 0, 0, state.width, state.height);
    cameraCtx.restore();
  }

  // Clear UI overlay
  uiCtx.clearRect(0, 0, state.width, state.height);

  // Process hand landmarks
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    const results = state.handLandmarker.detectForVideo(video, now);

    if (results.landmarks && results.landmarks.length > 0) {
      const landmarks = results.landmarks[0];
      const rawGesture = detectGesture(landmarks);
      const gesture = stabilizeGesture(rawGesture);

      if (!state.isModalOpen) {
        // Handle interactions
        if (gesture === 'index_finger') handleDrawing(landmarks);
        if (gesture === 'open_palm') handleErasing(landmarks);
        if (gesture === 'pinch') handleGrab(landmarks);
        
        // Finalize any in-progress stroke if not drawing
        if (gesture !== 'index_finger' && state.currentStroke && state.currentStroke.points.length > 1) {
          finalizeStroke(state.currentStroke);
          state.currentStroke = null;
          redrawStrokes();
        }
      }

      // Render hand overlay
      drawHandSkeleton(uiCtx, landmarks);
      drawCursorIndicator(uiCtx, landmarks, gesture);
      if (state.colorPicker.active) drawColorWheel(uiCtx);
    } else {
      // No hand detected
      if (state.currentGesture !== 'none') {
        onGestureChange(state.currentGesture, 'none');
        state.currentGesture = 'none';
      }
      if (state.currentStroke && state.currentStroke.points.length > 1) {
        finalizeStroke(state.currentStroke);
        state.currentStroke = null;
        redrawStrokes();
      }
      if (state.colorPicker.active) {
        state.colorPicker.active = false;
        state.pinchMode = null;
      }
    }
  }

  // Update particles
  updateAndDrawParticles(uiCtx);

  requestAnimationFrame(renderLoop);
}

// ── UI Event Handlers ──────────────────────────────────────────

// Color palette
document.querySelectorAll('.color-swatch').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.color-swatch').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.activeColor = btn.dataset.color;
    playTone(1000, 0.05, 'sine', 0.03);
  });
});

// Thickness
thicknessSlider.addEventListener('input', () => {
  state.thickness = parseInt(thicknessSlider.value);
  thicknessValue.textContent = `${state.thickness}px`;
});

// Glow
glowSlider.addEventListener('input', () => {
  state.glowIntensity = parseInt(glowSlider.value);
  glowValue.textContent = `${state.glowIntensity}%`;
});

// Undo
$('btn-undo').addEventListener('click', () => {
  if (state.strokes.length > 0) {
    state.strokes.pop();
    redrawStrokes();
    playTone(500, 0.08, 'sine', 0.03);
  }
});

// Clear
$('btn-clear').addEventListener('click', () => {
  state.strokes = [];
  state.currentStroke = null;
  state.particles = [];
  redrawStrokes();
  playTone(300, 0.15, 'triangle', 0.04);
});

// Camera toggle — cycles: Camera ON → Camera Dim → Dark Canvas
$('btn-camera-toggle').addEventListener('click', () => {
  if (state.showCamera && state.cameraOpacity > 0.2) {
    // Currently full → dim
    state.cameraOpacity = 0.15;
    cameraModeText.textContent = 'Camera DIM';
    cameraModeIndicator.classList.remove('dark-mode');
  } else if (state.showCamera && state.cameraOpacity <= 0.2) {
    // Currently dim → off
    state.showCamera = false;
    state.cameraOpacity = 0;
    cameraModeText.textContent = 'Dark Canvas';
    cameraModeIndicator.classList.add('dark-mode');
    $('btn-camera-toggle').classList.remove('active');
  } else {
    // Currently off → full
    state.showCamera = true;
    state.cameraOpacity = 0.35;
    cameraModeText.textContent = 'Camera ON';
    cameraModeIndicator.classList.remove('dark-mode');
    $('btn-camera-toggle').classList.add('active');
  }
  playModeSwitch();
});

// Also allow clicking the indicator to toggle
cameraModeIndicator.addEventListener('click', () => {
  $('btn-camera-toggle').click();
});

// Save
$('btn-save').addEventListener('click', () => {
  // Composite all canvases
  const exportCanvas = document.createElement('canvas');
  exportCanvas.width = state.width;
  exportCanvas.height = state.height;
  const exportCtx = exportCanvas.getContext('2d');

  // Dark background
  exportCtx.fillStyle = '#07070d';
  exportCtx.fillRect(0, 0, state.width, state.height);

  // Drawing layer
  exportCtx.drawImage(drawingCanvas, 0, 0);

  const link = document.createElement('a');
  link.download = `air-draw-${Date.now()}.png`;
  link.href = exportCanvas.toDataURL('image/png');
  link.click();

  playTone(800, 0.1, 'sine', 0.04);
});

// Onboarding start
btnStart.addEventListener('click', () => {
  onboardingModal.classList.add('hidden');
  state.isModalOpen = false;
  playTone(800, 0.1, 'sine', 0.04);
  
  // Reset HUD
  updateGestureHUD('idle');
});

// ── Initialization ─────────────────────────────────────────────
async function init() {
  resizeCanvases();

  try {
    // Load MediaPipe and webcam in parallel
    const [mpReady] = await Promise.all([
      initMediaPipe(),
      initWebcam()
    ]);

    state.isReady = true;

    // Complete the loader animation
    const loaderFill = document.querySelector('.loader-bar-fill');
    loaderFill.style.animation = 'none';
    loaderFill.style.width = '100%';
    loaderFill.style.transition = 'width 0.4s ease';

    // Fade out loading screen
    setTimeout(() => {
      loadingScreen.classList.add('fade-out');
      appEl.classList.remove('hidden');
      onboardingModal.classList.remove('hidden');
    }, 600);

    // Remove loading screen after fade
    setTimeout(() => {
      loadingScreen.style.display = 'none';
    }, 1200);

    // Start render loop
    renderLoop();

  } catch (error) {
    console.error('Failed to initialize Air Draw:', error);
    document.querySelector('.loader-subtitle').textContent = 
      'Error: Camera access required. Please allow camera permissions and reload.';
    document.querySelector('.loader-subtitle').style.color = '#ff2d6b';
    document.querySelector('.loader-bar').style.display = 'none';
  }
}

init();
