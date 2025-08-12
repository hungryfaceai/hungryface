import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");

let poseLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let drawPolygonBtn;
let clearPolygonBtn;
let webcamRunning = false;
const videoHeight = "360px";
const videoWidth = "480px";

// Anti-flicker smoothing: consecutive frames needed to switch state
const OUTSIDE_THRESHOLD_FRAMES = 5;
const INSIDE_THRESHOLD_FRAMES = 5;
let outsideFrames = 0;
let insideFrames = 0;
let isOutsideState = false; // current debounced state

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

/**************** Demo 1: click-on-image ****************/
const imageContainers = document.getElementsByClassName("detectOnClick");
for (let i = 0; i < imageContainers.length; i++) {
  imageContainers[i].children[0].addEventListener("click", handleClick);
}
async function handleClick(event) {
  if (!poseLandmarker) return;
  if (runningMode === "VIDEO") {
    runningMode = "IMAGE";
    await poseLandmarker.setOptions({ runningMode: "IMAGE" });
  }
  const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
  for (let i = allCanvas.length - 1; i >= 0; i--) {
    const n = allCanvas[i];
    n.parentNode.removeChild(n);
  }

  poseLandmarker.detect(event.target, (result) => {
    const canvas = document.createElement("canvas");
    canvas.setAttribute("class", "canvas");
    canvas.width = event.target.naturalWidth;
    canvas.height = event.target.naturalHeight;
    canvas.style = `left:0;top:0;width:${event.target.width}px;height:${event.target.height}px;`;

    event.target.parentNode.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    const drawingUtils = new DrawingUtils(ctx);

    for (const landmark of result.landmarks) {
      drawingUtils.drawLandmarks(landmark, {
        radius: (data) =>
          DrawingUtils.lerp((data.from && data.from.z) ?? 0, -0.15, 0.1, 5, 1)
      });
      drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
    }
  });
}

/**************** Demo 2: webcam ****************************/
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const drawingUtils = new DrawingUtils(canvasCtx);

const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  drawPolygonBtn = document.getElementById("drawPolygonBtn");
  clearPolygonBtn = document.getElementById("clearPolygonBtn");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

/************ Polygon drawing state & helpers *************/
let polygonPoints = [];       // [{x,y} in canvas pixels]
let polygonClosed = false;
let drawingMode = false;

// Because the canvas is horizontally flipped via CSS, adjust click X.
function getCanvasCoordsFromEvent(e) {
  const rect = canvasElement.getBoundingClientRect();
  const xVis = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  // Map CSS pixels to canvas pixels
  const scaleX = canvasElement.width / rect.width;
  const scaleY = canvasElement.height / rect.height;

  // Compensate for rotateY(180deg) on .output_canvas (mirror X)
  const x = (rect.width - xVis) * scaleX;
  return { x, y: y * scaleY };
}

function drawPolygonOverlay(ctx) {
  if (polygonPoints.length === 0) return;
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#00BCD4";
  ctx.fillStyle = "rgba(0, 188, 212, 0.1)";

  ctx.beginPath();
  ctx.moveTo(polygonPoints[0].x, polygonPoints[0].y);
  for (let i = 1; i < polygonPoints.length; i++) {
    ctx.lineTo(polygonPoints[i].x, polygonPoints[i].y);
  }
  if (polygonClosed) ctx.closePath();
  ctx.stroke();
  if (polygonClosed) ctx.fill();

  // draw handles
  for (const p of polygonPoints) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#00BCD4";
    ctx.fill();
  }
  ctx.restore();
}

// Ray-casting point-in-polygon
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

function enableDrawing() {
  drawingMode = !drawingMode;
  drawPolygonBtn.querySelector(".mdc-button__label").innerText =
    drawingMode ? "FINISH DRAWING" : "DRAW ZONE";
  if (!drawingMode && polygonPoints.length >= 3) {
    polygonClosed = true;
  }
}

function clearPolygon() {
  polygonPoints = [];
  polygonClosed = false;
  drawingMode = false;
  drawPolygonBtn.querySelector(".mdc-button__label").innerText = "DRAW ZONE";
  // reset debounce state
  outsideFrames = 0;
  insideFrames = 0;
  isOutsideState = false;
  document.body.style.backgroundColor = ""; // reset bg
}

// Mouse interactions on the canvas
canvasElement.addEventListener("click", (e) => {
  if (!drawingMode) return;
  const p = getCanvasCoordsFromEvent(e);
  polygonPoints.push(p);
});

canvasElement.addEventListener("dblclick", () => {
  if (!drawingMode) return;
  if (polygonPoints.length >= 3) {
    polygonClosed = true;
    drawingMode = false;
    drawPolygonBtn.querySelector(".mdc-button__label").innerText = "DRAW ZONE";
  }
});

canvasElement.addEventListener("contextmenu", (e) => {
  // Right-click to undo last point while drawing
  if (!drawingMode) return;
  e.preventDefault();
  polygonPoints.pop();
});

drawPolygonBtn?.addEventListener("click", enableDrawing);
clearPolygonBtn?.addEventListener("click", clearPolygon);

/**************** Webcam start/loop *****************/
function enableCam() {
  if (!poseLandmarker) {
    console.log("Wait! poseLandmarker not loaded yet.");
    return;
  }

  webcamRunning = !webcamRunning;
  enableWebcamButton.querySelector(".mdc-button__label").innerText =
    webcamRunning ? "DISABLE PREDICTIONS" : "ENABLE WEBCAM";

  const constraints = { video: true };
  navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
    video.srcObject = stream;
    video.addEventListener("loadeddata", predictWebcam);
  });
}

let lastVideoTime = -1;
async function predictWebcam() {
  canvasElement.style.height = videoHeight;
  video.style.height = videoHeight;
  canvasElement.style.width = videoWidth;
  video.style.width = videoWidth;

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

      // Draw pose & compute "any landmark outside" flag
      let anyOutside = false;
      for (const landmark of result.landmarks) {
        // Draw the pose first
        drawingUtils.drawLandmarks(landmark, {
          radius: (data) =>
            DrawingUtils.lerp((data.from && data.from.z) ?? 0, -0.15, 0.1, 5, 1)
        });
        drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);

        if (polygonClosed && polygonPoints.length >= 3) {
          // Test every landmark; if any is outside, flag it
          for (const p of landmark) {
            const x = p.x * canvasElement.width;
            const y = p.y * canvasElement.height;
            if (!pointInPolygon({ x, y }, polygonPoints)) {
              anyOutside = true;
              break;
            }
          }
        }
      }

      // Debounce / anti-flicker state machine
      if (polygonClosed && polygonPoints.length >= 3) {
        if (anyOutside) {
          outsideFrames++;
          insideFrames = 0;
          if (!isOutsideState && outsideFrames >= OUTSIDE_THRESHOLD_FRAMES) {
            isOutsideState = true;
            document.body.style.backgroundColor = "rgba(255,0,0,0.25)";
          }
        } else {
          insideFrames++;
          outsideFrames = 0;
          if (isOutsideState && insideFrames >= INSIDE_THRESHOLD_FRAMES) {
            isOutsideState = false;
            document.body.style.backgroundColor = "";
          }
        }
      } else {
        // No polygon → reset
        outsideFrames = 0;
        insideFrames = 0;
        isOutsideState = false;
        document.body.style.backgroundColor = "";
      }

      // Draw polygon overlay last so it stays on top
      drawPolygonOverlay(canvasCtx);
      canvasCtx.restore();
    });
  }

  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}
