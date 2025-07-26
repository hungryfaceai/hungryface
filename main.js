// Copyright 2025 The HungryFace Authors.
// Licensed under the Apache License, Version 2.0

import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

let faceLandmarker;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;

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

  function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
  } else {
    console.warn("getUserMedia() is not supported by your browser");
  }

  function enableCam() {
    if (!faceLandmarker) {
      console.log("Wait! faceLandmarker not loaded yet.");
      return;
    }

    webcamRunning = !webcamRunning;
    enableWebcamButton.querySelector(".mdc-button__label").innerText =
      webcamRunning ? "DISABLE PREDICTIONS" : "ENABLE WEBCAM";

    const constraints = { video: true };

    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
      video.srcObject = stream;
      video.addEventListener("loadeddata", () => {
        // Match sizes after webcam is ready
        const width = video.videoWidth;
        const height = video.videoHeight;

        video.width = width;
        video.height = height;
        canvasElement.width = width;
        canvasElement.height = height;

        video.style.width = width + "px";
        video.style.height = height + "px";
        canvasElement.style.width = width + "px";
        canvasElement.style.height = height + "px";

        predictWebcam();
      });
    });
  }

  let lastVideoTime = -1;
  let results = undefined;

  async function predictWebcam() {
    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    // Resize canvas if needed
    canvasElement.width = videoWidth;
    canvasElement.height = videoHeight;

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
          radius: 1.0 // Matches Python cv2.circle radius=1
        });
      }
    }

    drawBlendShapes(videoBlendShapes, results.faceBlendshapes);

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
});
