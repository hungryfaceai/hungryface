// vision-task.js — create a MediaPipe vision task, GPU first with a CPU fallback.
//
// MediaPipe's GPU delegate needs WebGL2. On devices/browsers without it (some older phones,
// locked-down or low-power modes, some in-app browsers) createFromOptions throws, and the
// detection mode simply never starts. This retries once on the CPU, which is slower but works.
//
// Usage:
//   import { createVisionTask } from '/hungryface/webrtc/shared/vision-task.js';
//   poseTask = await createVisionTask(PoseLandmarker, vision, {
//     baseOptions: { modelAssetPath: URL },   // delegate is chosen here
//     runningMode: 'VIDEO', numPoses: 1,
//   }, { label: 'motion' });

const lastDelegate = new Map(); // label -> 'GPU' | 'CPU' (for diagnostics)

export async function createVisionTask(TaskClass, fileset, options, { label = 'vision', log = console } = {}) {
  const withDelegate = (delegate) => ({
    ...options,
    baseOptions: { ...(options.baseOptions || {}), delegate },
  });
  try {
    const task = await TaskClass.createFromOptions(fileset, withDelegate('GPU'));
    lastDelegate.set(label, 'GPU');
    return task;
  } catch (gpuErr) {
    log.warn?.(`[${label}] GPU delegate unavailable, falling back to CPU:`, gpuErr?.message || gpuErr);
    const task = await TaskClass.createFromOptions(fileset, withDelegate('CPU')); // throws if CPU fails too
    lastDelegate.set(label, 'CPU');
    return task;
  }
}

/** Which delegate the last task created under `label` is using ('GPU', 'CPU', or undefined). */
export function visionDelegate(label) {
  return lastDelegate.get(label);
}
