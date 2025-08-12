import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");

/* ---------- UI refs ---------- */
let poseLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let drawPolygonBtn;
let clearPolygonBtn;
let startCameraBtn;
let cameraSelect;
let webcamRunning = false;

const stage = document.getElementById("stage");
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

/* ---------- Mobile detection + camera helpers ---------- */
function isIOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

let currentStream = null;
let isMirrored = true; // mirror front camera by default

function getCameraConstraints() {
  const selected = cameraSelect.value;

  if (isIOS()) {
    // iOS: prefer facingMode because deviceId enumeration is unreliable
    const facingMode = selected.includes("environment") ? { exact: "environment" } : "user";
    return { video: { facingMode } };
  }

  // Other platforms: use deviceId if provided
  return selected
    ? { video: { deviceId: { exact: selected } } }
    : { video: true };
}

function inferIsFrontFromSelection(constraints) {
  if (constraints?.video && "facingMode" in constraints.video) {
    return constraints.video.facingMode === "user";
  }
  // For deviceId path, try to infer from option label
  const opt = cameraSelect.selectedOptions[0];
  const label = (opt?.textContent || "").toLowerCase();
  if (label.includes("front")) return true;
  if (label.includes("back") || label.includes("rear") || label.includes("environment")) return false;
  // Fallback: assume front when uncertain
  return true;
}

function applyVideoMirroring(isFront) {
  isMirrored = !!isFront;
  const t = isMirrored ? "scaleX(-1)" : "none";
  video.style.transform = t;
  canvasElement.style.transform = t;
}

async function stopCurrentStream() {
  try {
    currentStream?.getTracks()?.forEach(t => t.stop());
  } catch {}
  currentStream = null;
}

/* ---------- Smoothing + alert ---------- */
const OUTSIDE_THRESHOLD_FRAMES = 5;
const INSIDE_THRESHOLD_FRAMES = 5;
let outsideFrames = 0;
let insideFrames = 0;
let isOutsideState = false; // debounced
// Match your preferred red; tweak alpha for strength
const ALERT_BG = "rgba(157, 0, 0, 0.86)";

/* ---------- Responsive canvas ---------- */
let lastCanvasW = 0;
let lastCanvasH = 0;
let polygonInitialized = false;

const ro = new ResizeObserver(() => resizeToStage());
ro.observe(stage);

function resizeToStage() {
  const rect = stage.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const newW = Math.max(1, Math.round(rect.width * dpr));
  const newH = Math.max(1, Math.round(rect.height * dpr));
  if (canvasElement.width === newW && canvasElement.height === newH) return;

  // Scale polygon to new size to preserve shape
  const scaleX = lastCanvasW ? newW / lastCanvasW : 1;
  const scaleY = lastCanvasH ? newH / lastCanvasH : 1;
  if (polygonInitialized && (scaleX !== 1 || scaleY !== 1)) {
    for (const p of polygonPoints) {
      p.x *= scaleX; p.y *= scaleY;
    }
  }

  canvasElement.width = newW;
  canvasElement.height = newH;
  lastCanvasW = newW;
  lastCanvasH = newH;

  if (!polygonInitialized && newW > 0 && newH > 0) {
    setDefaultPolygon();
    polygonInitialized = true;
  }
}

/* ---------- Polygon state (draggable) ---------- */
let polygonPoints = [];       // [{x,y} in canvas pixels]
let polygonClosed = false;
let drawingMode = false;
let draggingPointIndex = -1;
let hoveredPointIndex = -1;
let isDragging = false;

function setDefaultPolygon() {
  polygonPoints = [
    { x: 0, y: 0 },
    { x: canvasElement.width, y: 0 },
    { x: canvasElement.width, y: canvasElement.height },
    { x: 0, y: canvasElement.height }
  ];
  polygonClosed = true;
  drawingMode = false;
}

function clampToCanvas(pt) {
  return {
    x: Math.max(0, Math.min(canvasElement.width, pt.x)),
    y: Math.max(0, Math.min(canvasElement.height, pt.y))
  };
}

function getCanvasCoordsFromEvent(e) {
  const rect = canvasElement.getBoundingClientRect();
  const xVis = e.clientX - rect.left;
  const yVis = e.clientY - rect.top;
  const scaleX = canvasElement.width / rect.width;
  const scaleY = canvasElement.height / rect.height;
  const x = isMirrored ? (rect.width - xVis) * scaleX : xVis * scaleX;
  const y = yVis * scaleY;
  return { x, y };
}

function getHitRadiusCanvasPx() {
  const rect = canvasElement.getBoundingClientRect();
  const scaleX = canvasElement.width / rect.width;
  const dpr = window.devicePixelRatio || 1;
  return 10 * Math.max(1, scaleX / dpr) * dpr; // ~10 CSS px in canvas px
}

function findHandleIndexAt(pos) {
  const r2 = getHitRadiusCanvasPx() ** 2;
  let best = -1, bestDist = Infinity;
  for (let i = 0; i < polygonPoints.length; i++) {
    const dx = polygonPoints[i].x - pos.x;
    const dy = polygonPoints[i].y - pos.y;
    const d2 = dx*dx + dy*dy;
    if (d2 <= r2 && d2 < bestDist) { best = i; bestDist = d2; }
  }
  return best;
}

function drawPolygonOverlay(ctx) {
  if (!polygonPoints.length) return;
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#00BCD4";
  ctx.fillStyle = "rgba(0, 188, 212, 0.1)";
  ctx.beginPath();
  ctx.moveTo(polygonPoints[0].x, polygonPoints[0].y);
  for (let i = 1; i < polygonPoints.length; i++) ctx.lineTo(polygonPoints[i].x, polygonPoints[i].y);
  if (polygonClosed) ctx.closePath();
  ctx.stroke();
  if (polygonClosed) ctx.fill();
  // handles
  for (let i = 0; i < polygonPoints.length; i++) {
    const p = polygonPoints[i];
    ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#00BCD4"; ctx.fill();
  }
  // highlight
  const idx = isDragging ? draggingPointIndex : hoveredPointIndex;
  if (idx >= 0 && idx < polygonPoints.length) {
    const p = polygonPoints[idx];
    ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
    ctx.strokeStyle = "#FF9800"; ctx.lineWidth = 2; ctx.stroke();
  }
  ctx.restore();
}

// Draw readable overlay text (counter mirror only when needed)
function drawLimbListOverlay(ctx, lines) {
  if (!lines?.length) return;
  ctx.save();
  if (isMirrored) { // counter the CSS transform so text isn't flipped
    ctx.translate(canvasElement.width, 0);
    ctx.scale(-1, 1);
  }
  const padding = 8;
  const lh = 16;
  ctx.font = "14px sans-serif";
  const title = "Outside zone";
  let maxW = ctx.measureText(title).width;
  for (const l of lines) maxW = Math.max(maxW, ctx.measureText("• " + l).width);
  const boxW = maxW + padding * 2;
  const boxH = padding * 2 + lh * (lines.length + 1);
  const x = 10, y = 10;
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.fillRect(x, y, boxW, boxH);
  ctx.fillStyle = "#fff";
  ctx.textBaseline = "top";
  ctx.fillText(title, x + padding, y + padding);
  let cy = y + padding + lh;
  for (const l of lines) { ctx.fillText("• " + l, x + padding, cy); cy += lh; }
  ctx.restore();
}

// Point-in-polygon (ray cast)
function pointInPolygon(point, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y;
    const xj = poly[j].x, yj = poly[j].y;
    const intersect = ((yi > point.y) !== (yj > point.y)) &&
      (point.x < ((xj - xi) * (point.y - yi)) / (yj - yi + 1e-9) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

/* ---------- Limb definitions (BlazePose indices) ---------- */
const LM = {
  LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_WRIST: 15, RIGHT_WRIST: 16,
  LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_KNEE: 25, RIGHT_KNEE: 26,
  LEFT_ANKLE: 27, RIGHT_ANKLE: 28
};
const LIMBS = [
  { name: "Left Upper Arm", a: LM.LEFT_SHOULDER, b: LM.LEFT_ELBOW },
  { name: "Left Forearm", a: LM.LEFT_ELBOW, b: LM.LEFT_WRIST },
  { name: "Right Upper Arm", a: LM.RIGHT_SHOULDER, b: LM.RIGHT_ELBOW },
  { name: "Right Forearm", a: LM.RIGHT_ELBOW, b: LM.RIGHT_WRIST },
  { name: "Left Thigh", a: LM.LEFT_HIP, b: LM.LEFT_KNEE },
  { name: "Left Shin", a: LM.LEFT_KNEE, b: LM.LEFT_ANKLE },
  { name: "Right Thigh", a: LM.RIGHT_HIP, b: LM.RIGHT_KNEE },
  { name: "Right Shin", a: LM.RIGHT_KNEE, b: LM.RIGHT_ANKLE }
];

/* ---------- MediaPipe setup ---------- */
const createPoseLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU"
    },
    runningMode,
    numPoses: 1
  });
  demosSection.classList.remove("invisible");
};
createPoseLandmarker();

/* ---------- Camera listing ---------- */
async function listCameras() {
  cameraSelect.innerHTML = "";
  if (isIOS()) {
    // iOS: provide robust facingMode choices
    cameraSelect.innerHTML = `
      <option value="user">Front (iOS)</option>
      <option value="environment">Back (iOS)</option>
    `;
    return;
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const vids = devices.filter(d => d.kind === "videoinput");
    if (!vids.length) {
      cameraSelect.innerHTML = `<option value="">Default Camera</option>`;
      return;
    }
    for (const dev of vids) {
      const opt = document.createElement("option");
      opt.value = dev.deviceId;
      // After permissions, labels usually include "front"/"back"
      opt.text = dev.label || `Camera ${cameraSelect.length + 1}`;
      cameraSelect.appendChild(opt);
    }
  } catch {
    cameraSelect.innerHTML = `<option value="">Default Camera</option>`;
  }
}

/* ---------- Start / switch camera ---------- */
async function startOrSwitchCamera() {
  const constraints = getCameraConstraints();
  const isFront = inferIsFrontFromSelection(constraints);

  await stopCurrentStream();

  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  currentStream = stream;
  video.srcObject = stream;

  // When metadata is ready, set aspect ratio & resize canvas
  await new Promise(resolve => {
    const onLoaded = () => { video.removeEventListener("loadedmetadata", onLoaded); resolve(); };
    video.addEventListener("loadedmetadata", onLoaded, { once: true });
  });

  if (video.videoWidth && video.videoHeight) {
    stage.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  }
  resizeToStage();

  applyVideoMirroring(isFront);
}

/* ---------- Predict loop ---------- */
let lastVideoTime = -1;

async function predictWebcam() {
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await poseLandmarker.setOptions({ runningMode: "VIDEO" });
  }

  const startTimeMs = performance.now();
  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

      let anyOutside = false;
      let outsideLimbNames = [];

      for (const landmark of result.landmarks) {
        drawingUtils.drawLandmarks(landmark, {
          radius: (data) =>
            DrawingUtils.lerp((data.from && data.from.z) ?? 0, -0.15, 0.1, 5, 1)
        });
        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);

        if (polygonClosed && polygonPoints.length >= 3) {
          // Per-limb outside list: both endpoints must be inside; else outside
          for (const limb of LIMBS) {
            const pa = landmark[limb.a], pb = landmark[limb.b];
            if (!pa || !pb) continue;
            const ax = pa.x * canvasElement.width;
            const ay = pa.y * canvasElement.height;
            const bx = pb.x * canvasElement.width;
            const by = pb.y * canvasElement.height;
            const aIn = pointInPolygon({ x: ax, y: ay }, polygonPoints);
            const bIn = pointInPolygon({ x: bx, y: by }, polygonPoints);
            if (!(aIn && bIn)) outsideLimbNames.push(limb.name);
          }
          // Any landmark outside → background alert (debounced)
          for (const p of landmark) {
            const x = p.x * canvasElement.width;
            const y = p.y * canvasElement.height;
            if (!pointInPolygon({ x, y }, polygonPoints)) { anyOutside = true; break; }
          }
        }
      }

      // Debounce background
      if (polygonClosed && polygonPoints.length >= 3) {
        if (anyOutside) {
          outsideFrames++; insideFrames = 0;
          if (!isOutsideState && outsideFrames >= OUTSIDE_THRESHOLD_FRAMES) {
            isOutsideState = true; document.body.style.backgroundColor = ALERT_BG;
          }
        } else {
          insideFrames++; outsideFrames = 0;
          if (isOutsideState && insideFrames >= INSIDE_THRESHOLD_FRAMES) {
            isOutsideState = false; document.body.style.backgroundColor = "";
          }
        }
      } else {
        outsideFrames = insideFrames = 0; isOutsideState = false;
        document.body.style.backgroundColor = "";
        outsideLimbNames = [];
      }

      drawPolygonOverlay(canvasCtx);
      drawLimbListOverlay(canvasCtx, Array.from(new Set(outsideLimbNames)));

      canvasCtx.restore();
    });
  }

  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}

/* ---------- Drawing mode buttons ---------- */
function enableDrawing() {
  if (!drawingMode) {
    // Enter drawing mode → clear polygon and reset state
    polygonPoints = [];
    polygonClosed = false;
    drawingMode = true;
    hoveredPointIndex = -1; draggingPointIndex = -1; isDragging = false;
    outsideFrames = 0; insideFrames = 0; isOutsideState = false;
    document.body.style.backgroundColor = "";
    drawPolygonBtn.querySelector(".mdc-button__label").innerText = "FINISH DRAWING";
  } else {
    drawingMode = false;
    if (polygonPoints.length >= 3) polygonClosed = true;
    drawPolygonBtn.querySelector(".mdc-button__label").innerText = "DRAW ZONE";
  }
}

function clearPolygon() {
  setDefaultPolygon();
  outsideFrames = insideFrames = 0; isOutsideState = false;
  document.body.style.backgroundColor = "";
}

/* ---------- Events ---------- */
function attachPointerEvents() {
  canvasElement.addEventListener("click", (e) => {
    if (!drawingMode) return;
    const p = getCanvasCoordsFromEvent(e);
    polygonPoints.push(p);
  });

  canvasElement.addEventListener("dblclick", () => {
    if (!drawingMode) return;
    if (polygonPoints.length >= 3) {
      polygonClosed = true; drawingMode = false;
      drawPolygonBtn.querySelector(".mdc-button__label").innerText = "DRAW ZONE";
    }
  });

  canvasElement.addEventListener("contextmenu", (e) => {
    if (!drawingMode) return;
    e.preventDefault(); polygonPoints.pop();
  });

  canvasElement.addEventListener("mousedown", (e) => {
    if (drawingMode) return;
    const pos = getCanvasCoordsFromEvent(e);
    const idx = findHandleIndexAt(pos);
    if (idx !== -1) { isDragging = true; draggingPointIndex = idx; }
  });

  window.addEventListener("mousemove", (e) => {
    const pos = getCanvasCoordsFromEvent(e);
    if (isDragging && draggingPointIndex !== -1) {
      polygonPoints[draggingPointIndex] = clampToCanvas(pos);
    } else if (!drawingMode) {
      hoveredPointIndex = findHandleIndexAt(pos);
    }
  });

  window.addEventListener("mouseup", () => {
    isDragging = false; draggingPointIndex = -1;
  });
}

/* ---------- Init ---------- */
document.addEventListener("DOMContentLoaded", async () => {
  enableWebcamButton = document.getElementById("webcamButton");
  drawPolygonBtn = document.getElementById("drawPolygonBtn");
  clearPolygonBtn = document.getElementById("clearPolygonBtn");
  startCameraBtn = document.getElementById("startCamera");
  cameraSelect = document.getElementById("cameraSelect");

  attachPointerEvents();
  await listCameras();

  // Start/Switch camera
  startCameraBtn.addEventListener("click", async () => {
    try { await startOrSwitchCamera(); }
    catch (err) { console.error("Could not access selected camera:", err); }
  });

  // If user changes selection while streaming, auto switch
  cameraSelect.addEventListener("change", async () => {
    if (currentStream) {
      try { await startOrSwitchCamera(); }
      catch (err) { console.error("Switch camera failed:", err); }
    }
  });

  // Enable/disable predictions loop (will auto-start camera if none)
  enableWebcamButton.addEventListener("click", async () => {
    if (!poseLandmarker) { console.log("Wait! poseLandmarker not loaded yet."); return; }

    // If no stream yet, start with current selection first
    if (!currentStream) {
      try { await startOrSwitchCamera(); } catch (e) {
        console.error("Camera start failed:", e); return;
      }
    }

    webcamRunning = !webcamRunning;
    enableWebcamButton.querySelector(".mdc-button__label").innerText =
      webcamRunning ? "DISABLE PREDICTIONS" : "ENABLE PREDICTIONS";

    if (webcamRunning) {
      predictWebcam();
    } else {
      // stopping loop is enough; stream stays active so user can switch cameras
    }
  });

  // Drawing buttons
  drawPolygonBtn.addEventListener("click", enableDrawing);
  clearPolygonBtn.addEventListener("click", clearPolygon);
});
