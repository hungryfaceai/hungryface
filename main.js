// Copyright 2025 The HungryFace Authors.
// Licensed under the Apache License, Version 2.0

import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

let expressionModel;
const classes = ["happy", "sad"]; // 🔁 Customize as needed

async function loadExpressionModel() {
  try {
    expressionModel = await tf.loadLayersModel("model_tfjs/model.json");
    console.log("✅ Expression model loaded OK.");
  } catch (err) {
    console.error("❌ Failed to load model:", err);
  }
}
loadExpressionModel();

// --- Prediction smoothing ---
const predictionBuffer = [];
const SMOOTHING_WINDOW = 10; // last N frames

function smoothPredictions(newPreds) {
  predictionBuffer.push(newPreds);
  if (predictionBuffer.length > SMOOTHING_WINDOW) {
    predictionBuffer.shift(); // remove oldest
  }

  const summed = Array(newPreds.length).fill(0);
  for (const preds of predictionBuffer) {
    preds.forEach((val, idx) => (summed[idx] += val));
  }

  const averaged = summed.map(val => val / predictionBuffer.length);
  return averaged;
}

let faceLandmarker;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoWidth = 1024;

document.addEventListener("DOMContentLoaded", () => {
  const demosSection = document.getElementById("demos");
  const videoBlendShapes = document.getElementById("video-blend-shapes");
  const video = document.getElementById("webcam");
  const canvasElement = document.getElementById("output_canvas");
  const canvasCtx = canvasElement.getContext("2d");

  const drawingUtils = new DrawingUtils(canvasCtx);

  async function createFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU"
      },
      outputFaceBlendshapes: true,
      runningMode,
      numFaces: 1
    });

    demosSection.classList.remove("invisible");
  }

  createFaceLandmarker();
listCameras(); //added this line

  function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
  } else {
    console.warn("getUserMedia() is not supported by your browser");
  }
  //added:
document.getElementById("startCamera").addEventListener("click", async () => {
  const deviceId = document.getElementById("cameraSelect").value;
  const constraints = {
    video: { deviceId: { exact: deviceId } }
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    const video = document.getElementById("webcam");
    video.srcObject = stream;
    video.play();
  } catch (err) {
    console.error("Error accessing camera:", err);
  }
});

//end added  

  function enableCam() {
    if (!faceLandmarker) {
      console.log("Wait! faceLandmarker not loaded yet.");
      return;
    }

    webcamRunning = !webcamRunning;
    enableWebcamButton.querySelector(".mdc-button__label").innerText =
      webcamRunning ? "DISABLE PREDICTIONS" : "ENABLE PREDICTIONS";

    const video = document.getElementById("webcam"); // ADDED
    const selectedDeviceId = document.getElementById("cameraSelect").value; // ADDED
	  
    //const constraints = { video: true }; //access the front camera
    const constraints = {
      video: { deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined } // 🔧 CHANGED
    };	  
/*navigator.mediaDevices.getUserMedia({
  video: { facingMode: { exact: "environment" } }
})
.then((stream) => {
  video.srcObject = stream;
  video.play();
})
.catch((err) => {
  console.error("Could not access back camera:", err);
});*/ //access the back camera
	

    /*navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
      video.srcObject = stream;
      video.addEventListener("loadeddata", predictWebcam);
    });
  }*/
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
      video.srcObject = stream;
      video.addEventListener("loadeddata", predictWebcam);
    }).catch((err) => { // 🔧 ADDED
      console.error("Could not access selected camera:", err); // 🔧 ADDED
    });
  }

  let lastVideoTime = -1;
  let results = undefined;

  async function predictWebcam() {
    const ratio = video.videoHeight / video.videoWidth;
	/*
    video.style.width = videoWidth + "px";
    video.style.height = videoWidth * ratio + "px";
	
    canvasElement.style.width = videoWidth + "px";
    canvasElement.style.height = videoWidth * ratio + "px";
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
	*/
const width = video.videoWidth;
const height = video.videoHeight;

// Match video display size to actual resolution
video.style.width = width + "px";
video.style.height = height + "px";

// Match canvas display size and internal pixel buffer
canvasElement.style.width = width + "px";
canvasElement.style.height = height + "px";
canvasElement.width = width;
canvasElement.height = height;


    if (runningMode === "IMAGE") {
      runningMode = "VIDEO";
      await faceLandmarker.setOptions({ runningMode });
    }

    const startTimeMs = performance.now();

    if (lastVideoTime !== video.currentTime) {
      lastVideoTime = video.currentTime;
      results = faceLandmarker.detectForVideo(video, startTimeMs);
    }

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (results.faceLandmarks) {
      for (const landmarks of results.faceLandmarks) {
        drawingUtils.drawLandmarks(landmarks, {
          color: "#00FF00",
          radius: 0.5
        });
      }
    }

    drawBlendShapes(videoBlendShapes, results.faceBlendshapes);

    // --- Run expression model with smoothing ---
    /*
	if (results.faceBlendshapes && results.faceBlendshapes.length > 0 && expressionModel) {
      const blendshapes = results.faceBlendshapes[0].categories.map(c => c.score);
      const inputTensor = tf.tensor(blendshapes, [1, 52, 1]); // or [1, 52] if needed

      const prediction = expressionModel.predict(inputTensor);
      prediction.array().then((preds) => {
        const smoothed = smoothPredictions(preds[0]);
        const maxIndex = smoothed.indexOf(Math.max(...smoothed));
        const label = classes[maxIndex];
        const confidence = smoothed[maxIndex];

        // --- Overlay label on canvas ---
        canvasCtx.font = "24px Arial";
        canvasCtx.fillStyle = "rgba(0, 0, 0, 0.6)";
        canvasCtx.fillRect(10, 10, 280, 36);
        canvasCtx.fillStyle = "#00FF00";
        canvasCtx.fillText(`Expression: ${label} (${(confidence * 100).toFixed(1)}%)`, 20, 36);
      });
    }
	*/
	if (results.faceBlendshapes && results.faceBlendshapes.length > 0 && expressionModel) {
  const blendshapes = results.faceBlendshapes[0].categories.map(c => c.score);
  const inputTensor = tf.tensor(blendshapes, [1, 52, 1]); // or [1, 52] if needed
  const prediction = expressionModel.predict(inputTensor);
  const preds = await prediction.data(); // await added here
  const smoothed = smoothPredictions(Array.from(preds));
  const maxIndex = smoothed.indexOf(Math.max(...smoothed));
  const label = classes[maxIndex];
  const confidence = smoothed[maxIndex];

//console.log("Running model prediction");
//console.log("Blendshapes:", blendshapes);
//console.log("Input tensor shape:", inputTensor.shape);

canvasCtx.font = "24px Arial";
canvasCtx.textBaseline = "top";
canvasCtx.fillStyle = "rgba(0, 0, 0, 0.6)";
canvasCtx.fillRect(10, 10, 320, 40); // background box

canvasCtx.fillStyle = "#00FF00";
console.log(`Expression: ${label} (${(confidence * 100).toFixed(1)}%)`);
canvasCtx.fillText(`Expression: ${label} (${(confidence * 100).toFixed(1)}%)`, 20, 20); // not working..
document.getElementById("expression-label").textContent =
  `DOM: ${label} (${(confidence * 100).toFixed(1)}%)`;

  //console.log("Raw model output:", preds);
//canvasCtx.fillStyle = "red"; //debug
//canvasCtx.font = "bold 24px monospace"; //debug
//canvasCtx.fillText("CANVAS DRAWING WORKS", 100, 100); //debug

  }


    if (webcamRunning) {
      window.requestAnimationFrame(predictWebcam);
    }
  }

  function drawBlendShapes(el, blendShapes) {
    if (!blendShapes || !blendShapes.length) return;

    let htmlMaker = "";
    blendShapes[0].categories.forEach((shape) => {
      htmlMaker += `
        <li class="blend-shapes-item">
          <span class="blend-shapes-label">${
            shape.displayName || shape.categoryName
          }</span>
          <span class="blend-shapes-value" style="width: calc(${
            +shape.score * 100
          }% - 120px)">${(+shape.score).toFixed(4)}</span>
        </li>
      `;
    });

    el.innerHTML = htmlMaker;
  }

//added:
  async function listCameras() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter(device => device.kind === "videoinput");

    const select = document.getElementById("cameraSelect");
	select.innerHTML = ""; // clear previous
    videoInputs.forEach(device => {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.text = device.label || `Camera ${select.length + 1}`;
      select.appendChild(option);
    });
  }
  
});
