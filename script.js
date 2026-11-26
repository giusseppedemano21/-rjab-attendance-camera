import {
    FaceDetector,
    FaceLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// =====================================
// MEDIAPIPE FACE DETECTOR
// =====================================

let faceDetector = null;
let faceDetectorReady = false;
let cameraReady = false;

// Separate VIDEO-mode detector for live face tracking.
// The original IMAGE-mode detector remains unchanged for photo-quality checks.
let liveFaceDetector = null;
let liveFaceDetectorReady = false;
let liveTrackingStarted = false;
let lastLiveDetectionAt = 0;
let noFaceFrames = 0;

// Optional AI diagnostic mode is issued by Apps Script only when the
// ATTENDANCE_AI_TEST_MODE script property is enabled. Normal attendance
// launches do not download or initialize the EdgeFace/MiniFASNet models.
const biometricAiTestMode = new URLSearchParams(window.location.search).get("aiTest") === "1";
const BIOMETRIC_AI_MODEL_CONFIG = Object.freeze({
    MODEL_VERSION: "EdgeFace-XXS",
    LANDMARKER_URL: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    ORT_WASM_URL: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/",
    EDGEFACE_URL: "./edgeface_xxs.onnx",
    PAD_V2_URL: "./MiniFASNetV2.onnx",
    PAD_V1SE_URL: "./MiniFASNetV1SE.onnx",
    EDGEFACE_SHA256: "dc674de4cbc77fa0bf9a82d5149558ab8581d82a2cd3bb60f28fd1a5d3ff8a2f",
    PAD_V2_SHA256: "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907",
    PAD_V1SE_SHA256: "ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676",
    EXPLORATORY_PAD_THRESHOLD: 0.80,
    PAD_SAMPLE_COUNT: 5,
    EMBEDDING_SAMPLE_COUNT: 3
});
let biometricAiReady = false;
let biometricAiBusy = false;
let biometricAiError = "";
let biometricLandmarker = null;
let biometricRecognitionSession = null;
let biometricPadV2Session = null;
let biometricPadV1seSession = null;
let biometricReferenceEmbedding = null; // Memory-only; cleared on reload or explicit clear.
let biometricAiVision = null;

function updateCameraReadyState() {

    if (cameraReady && faceDetectorReady && faceDetector) {

        captureBtn.hidden = false;
        captureBtn.disabled = false;

        status.innerHTML = "✅ Ready — Take Selfie";

        console.log("✅ Camera System Fully Ready");

    } else {

        captureBtn.hidden = true;
        captureBtn.disabled = true;

    }

}

async function initFaceDetector() {

    try {

        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );

        const MODEL =
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";

        // IMAGE detector: CPU lang (mas matatag sa lumang iPhone)
        faceDetector = await FaceDetector.createFromOptions(vision, {
            baseOptions: { modelAssetPath: MODEL, delegate: "CPU" },
            runningMode: "IMAGE",
            minDetectionConfidence: 0.5
        });

        console.log("✅ MediaPipe Face Detector Ready (CPU)");

        // LIVE tracker: huwag gawin sa iOS 16 pababa
        const isOldIOS = /iPhone OS (1[0-6])_/.test(navigator.userAgent);

        if (!isOldIOS) {

            try {

                liveFaceDetector = await FaceDetector.createFromOptions(vision, {
                    baseOptions: { modelAssetPath: MODEL, delegate: "CPU" },
                    runningMode: "VIDEO",
                    minDetectionConfidence: 0.5
                });

                liveFaceDetectorReady = true;
                console.log("✅ Live Face Tracker Ready");

            } catch (liveError) {

                liveFaceDetector = null;
                liveFaceDetectorReady = false;
                console.warn("⚠️ Live face tracking unavailable:", liveError);

            }

        }

        faceDetectorReady = true;

        updateCameraReadyState();
        startLiveFaceTracking();

        if (biometricAiTestMode) {
            const aiPanel = document.getElementById("biometricAiLab");
            if (aiPanel) aiPanel.hidden = false;
            initializeBiometricAITest(vision).catch(function (aiError) {
                biometricAiError = aiError && aiError.message ? aiError.message : String(aiError);
                setBiometricAiStatus("AI diagnostics could not initialize: " + biometricAiError);
                setBiometricAiResult("Normal selfie/GPS capture remains available. AI diagnostics are optional and are not part of attendance submission.");
            });
        }

    } catch (error) {

        console.error("❌ MediaPipe Face Detector failed:", error);

        faceDetectorReady = false;
        faceDetector = null;

        captureBtn.hidden = true;
        captureBtn.disabled = true;

        status.innerHTML =
            "⚠️ AI Scanner failed to load.<br><br>" +
            "Please reload the camera.";

    }
}
// =====================================
// OPTIONAL BIOMETRIC AI DIAGNOSTICS (TEST ONLY)
// EdgeFace embedding + two MiniFASNet PAD models.
// No image, embedding, PAD score, or identity result is sent to the portal.
// =====================================

function setBiometricAiStatus(message) {
    const el = document.getElementById("biometricAiModelStatus");
    if (el) el.textContent = message;
}

function setBiometricAiResult(message) {
    const el = document.getElementById("biometricAiResult");
    if (el) el.textContent = message;
}

function updateBiometricAiButtons() {
    if (!biometricAiTestMode) return;
    const live = !!(biometricAiReady && app && app.stream && video && !video.hidden && video.readyState >= 2 && video.videoWidth > 0);
    const ref = document.getElementById("biometricAiReferenceBtn");
    const compare = document.getElementById("biometricAiCompareBtn");
    const workerCompare = document.getElementById("biometricAiWorkerCompareBtn");
    const pad = document.getElementById("biometricAiPadBtn");
    const clear = document.getElementById("biometricAiClearBtn");
    if (ref) ref.disabled = !live || biometricAiBusy;
    if (compare) compare.disabled = !live || !biometricReferenceEmbedding || biometricAiBusy;
    if (workerCompare) workerCompare.disabled = !live || !biometricAiReady || biometricAiBusy;
    if (pad) pad.disabled = !live || biometricAiBusy;
    if (clear) clear.disabled = !biometricReferenceEmbedding || biometricAiBusy;
}

function loadOnnxRuntimeForBiometricTest() {
    if (window.ort && window.ort.InferenceSession) return Promise.resolve(window.ort);
    return new Promise(function (resolve, reject) {
        let script = document.querySelector('script[data-rjab-onnx-runtime="1"]');
        if (script) {
            script.addEventListener("load", function () { resolve(window.ort); }, { once: true });
            script.addEventListener("error", function () { reject(new Error("ONNX Runtime Web could not be downloaded.")); }, { once: true });
            return;
        }
        script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.min.js";
        script.async = true;
        script.dataset.rjabOnnxRuntime = "1";
        script.onload = function () {
            if (window.ort && window.ort.InferenceSession) resolve(window.ort);
            else reject(new Error("ONNX Runtime Web loaded but its API is unavailable."));
        };
        script.onerror = function () { reject(new Error("ONNX Runtime Web could not be downloaded. Check internet access.")); };
        document.head.appendChild(script);
    });
}

async function loadVerifiedBiometricOnnx(url, expectedSha, label) {
    const response = await fetch(url, { cache: "force-cache" });
    if (!response.ok) throw new Error("Missing " + label + " model (HTTP " + response.status + "). Upload the exact ONNX file beside index.html.");
    const bytes = new Uint8Array(await response.arrayBuffer());
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const actualSha = Array.from(new Uint8Array(digest), function (b) { return b.toString(16).padStart(2, "0"); }).join("");
    if (actualSha !== expectedSha) throw new Error(label + " model checksum mismatch. Do not use or replace the verified model with an unverified file.");
    return await window.ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all"
    });
}

async function initializeBiometricAITest(vision) {
    if (!biometricAiTestMode) return;
    if (biometricAiReady) return;
    setBiometricAiStatus("Loading Face Landmarker and ONNX Runtime. This is optional test mode…");
    setBiometricAiResult("Please keep the camera page open. The existing capture controls remain available.");

    const ort = await loadOnnxRuntimeForBiometricTest();
    ort.env.wasm.wasmPaths = BIOMETRIC_AI_MODEL_CONFIG.ORT_WASM_URL;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;

    biometricLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: BIOMETRIC_AI_MODEL_CONFIG.LANDMARKER_URL, delegate: "CPU" },
        runningMode: "VIDEO",
        numFaces: 2,
        outputFaceBlendshapes: false,
        minFaceDetectionConfidence: 0.60,
        minFacePresenceConfidence: 0.60,
        minTrackingConfidence: 0.60
    });

    setBiometricAiStatus("Loading EdgeFace-XXS recognition model…");
    biometricRecognitionSession = await loadVerifiedBiometricOnnx(
        BIOMETRIC_AI_MODEL_CONFIG.EDGEFACE_URL,
        BIOMETRIC_AI_MODEL_CONFIG.EDGEFACE_SHA256,
        "EdgeFace-XXS"
    );

    setBiometricAiStatus("Loading MiniFASNet PAD models…");
    biometricPadV2Session = await loadVerifiedBiometricOnnx(
        BIOMETRIC_AI_MODEL_CONFIG.PAD_V2_URL,
        BIOMETRIC_AI_MODEL_CONFIG.PAD_V2_SHA256,
        "MiniFASNet V2"
    );
    biometricPadV1seSession = await loadVerifiedBiometricOnnx(
        BIOMETRIC_AI_MODEL_CONFIG.PAD_V1SE_URL,
        BIOMETRIC_AI_MODEL_CONFIG.PAD_V1SE_SHA256,
        "MiniFASNet V1SE"
    );

    biometricAiReady = true;
    setBiometricAiStatus("Ready · EdgeFace-XXS + MiniFASNet V2/V1SE loaded and checksum-verified.");
    setBiometricAiResult("AI diagnostics are ready. Temporary reference and comparisons stay in this browser tab only.");
    updateBiometricAiButtons();
}

function biometricSleep(ms) {
    return new Promise(function (resolve) { setTimeout(resolve, ms); });
}

function getCurrentBiometricLandmarks() {
    if (!biometricAiReady || !biometricLandmarker) throw new Error("AI models are not ready yet.");
    if (!app || !app.stream || video.hidden || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
        throw new Error("The live camera is not active. Return to the camera preview and try again.");
    }
    const detection = biometricLandmarker.detectForVideo(video, performance.now());
    const faces = detection.faceLandmarks || [];
    if (faces.length !== 1) {
        throw new Error(faces.length === 0 ? "No face was detected. Center your face and try again." : "More than one face was detected. Only one person should be in the camera frame.");
    }
    if (!faces[0] || faces[0].length < 264) throw new Error("Face landmarks are incomplete. Reposition your face and try again.");
    return faces[0];
}

function biometricAveragePoints(indices, landmarks) {
    const points = indices.map(function (index) { return landmarks[index]; });
    return {
        x: points.reduce(function (sum, point) { return sum + point.x; }, 0) / points.length * video.videoWidth,
        y: points.reduce(function (sum, point) { return sum + point.y; }, 0) / points.length * video.videoHeight
    };
}

function biometricFivePointLandmarks(landmarks) {
    // Approximate five-point alignment adapted from the Phase 2B/2C prototype.
    // Validate alignment on RJAB device samples before treating scores as operational.
    return [
        biometricAveragePoints([33, 133], landmarks),
        biometricAveragePoints([362, 263], landmarks),
        biometricAveragePoints([1], landmarks),
        biometricAveragePoints([61], landmarks),
        biometricAveragePoints([291], landmarks)
    ];
}

function biometricSimilarityTransform(source, target) {
    const count = source.length;
    const srcMean = { x: 0, y: 0 };
    const dstMean = { x: 0, y: 0 };
    for (let i = 0; i < count; i++) {
        srcMean.x += source[i].x / count;
        srcMean.y += source[i].y / count;
        dstMean.x += target[i].x / count;
        dstMean.y += target[i].y / count;
    }
    let dot = 0, cross = 0, variance = 0;
    for (let i = 0; i < count; i++) {
        const sx = source[i].x - srcMean.x;
        const sy = source[i].y - srcMean.y;
        const dx = target[i].x - dstMean.x;
        const dy = target[i].y - dstMean.y;
        dot += sx * dx + sy * dy;
        cross += sx * dy - sy * dx;
        variance += sx * sx + sy * sy;
    }
    if (variance < 1e-6) throw new Error("Face landmarks are too close together. Reposition your face.");
    const a = dot / variance;
    const b = cross / variance;
    const tx = dstMean.x - (a * srcMean.x - b * srcMean.y);
    const ty = dstMean.y - (b * srcMean.x + a * srcMean.y);
    return { a: a, b: b, c: -b, d: a, e: tx, f: ty };
}

async function makeBiometricEmbedding(landmarks) {
    if (!biometricRecognitionSession) throw new Error("EdgeFace recognition model is not loaded.");
    const alignedCanvas = makeBiometricEmbedding.canvas || (makeBiometricEmbedding.canvas = document.createElement("canvas"));
    alignedCanvas.width = 112;
    alignedCanvas.height = 112;
    const ctx = alignedCanvas.getContext("2d", { willReadFrequently: true });
    const target = [
        { x: 38.2946, y: 51.6963 },
        { x: 73.5318, y: 51.5014 },
        { x: 56.0252, y: 71.7366 },
        { x: 41.5493, y: 92.3655 },
        { x: 70.7299, y: 92.2041 }
    ];
    const transform = biometricSimilarityTransform(biometricFivePointLandmarks(landmarks), target);
    ctx.setTransform(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, 112, 112);
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    const pixels = ctx.getImageData(0, 0, 112, 112).data;
    const plane = 112 * 112;
    const input = new Float32Array(3 * plane);
    for (let i = 0, p = 0; i < plane; i++) {
        const r = pixels[p++];
        const g = pixels[p++];
        const b = pixels[p++];
        p++;
        input[i] = (r - 127.5) / 127.5;
        input[plane + i] = (g - 127.5) / 127.5;
        input[2 * plane + i] = (b - 127.5) / 127.5;
    }

    const session = biometricRecognitionSession;
    const output = await session.run({ [session.inputNames[0]]: new window.ort.Tensor("float32", input, [1, 3, 112, 112]) });
    const raw = output[session.outputNames[0]].data;
    if (!raw || raw.length !== 512) throw new Error("EdgeFace returned an unexpected embedding size (expected 512 values).");
    const embedding = new Float32Array(raw.length);
    let norm = 0;
    for (let i = 0; i < raw.length; i++) norm += raw[i] * raw[i];
    norm = Math.sqrt(norm);
    if (!norm || !Number.isFinite(norm)) throw new Error("EdgeFace returned an invalid embedding.");
    for (let i = 0; i < raw.length; i++) embedding[i] = raw[i] / norm;
    return embedding;
}

async function captureBiometricTestEmbeddingSet(label) {
    const embeddings = [];
    for (let i = 0; i < BIOMETRIC_AI_MODEL_CONFIG.EMBEDDING_SAMPLE_COUNT; i++) {
        await biometricSleep(i === 0 ? 100 : 350);
        const landmarks = getCurrentBiometricLandmarks();
        setBiometricAiResult(label + " · processing sample " + (i + 1) + " of 3… Keep one face centered.");
        embeddings.push(await makeBiometricEmbedding(landmarks));
    }
    const average = new Float32Array(embeddings[0].length);
    for (const embedding of embeddings) {
        if (embedding.length !== average.length) throw new Error("Embedding dimension mismatch.");
        for (let i = 0; i < embedding.length; i++) average[i] += embedding[i] / embeddings.length;
    }
    let norm = 0;
    for (let i = 0; i < average.length; i++) norm += average[i] * average[i];
    norm = Math.sqrt(norm);
    if (!norm) throw new Error("Could not normalize the averaged face embedding.");
    for (let i = 0; i < average.length; i++) average[i] /= norm;
    return average;
}

function biometricCosineSimilarity(a, b) {
    if (!a || !b || a.length !== b.length) throw new Error("Temporary reference is missing or incompatible.");
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot;
}

function biometricClamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
}

function biometricMedian(values) {
    const sorted = values.slice().sort(function (a, b) { return a - b; });
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function makeBiometricPadTensor(scale, landmarks) {
    if (!landmarks || video.readyState < 2) throw new Error("A clear face must be visible before PAD analysis.");
    const xs = landmarks.map(function (point) { return point.x; });
    const ys = landmarks.map(function (point) { return point.y; });
    const bx = Math.min.apply(null, xs) * video.videoWidth;
    const by = Math.min.apply(null, ys) * video.videoHeight;
    const bw = (Math.max.apply(null, xs) - Math.min.apply(null, xs)) * video.videoWidth;
    const bh = (Math.max.apply(null, ys) - Math.min.apply(null, ys)) * video.videoHeight;
    if (bw < 40 || bh < 40) throw new Error("Face is too small for PAD analysis. Move a little closer.");

    const factor = Math.min((video.videoHeight - 1) / bh, (video.videoWidth - 1) / bw, scale);
    const cropW = bw * factor;
    const cropH = bh * factor;
    const cx = bx + bw / 2;
    const cy = by + bh / 2;
    let left = cx - cropW / 2;
    let top = cy - cropH / 2;
    let right = cx + cropW / 2;
    let bottom = cy + cropH / 2;
    if (left < 0) { right -= left; left = 0; }
    if (top < 0) { bottom -= top; top = 0; }
    if (right > video.videoWidth - 1) { left -= right - video.videoWidth + 1; right = video.videoWidth - 1; }
    if (bottom > video.videoHeight - 1) { top -= bottom - video.videoHeight + 1; bottom = video.videoHeight - 1; }
    left = biometricClamp(left, 0, video.videoWidth - 2);
    top = biometricClamp(top, 0, video.videoHeight - 2);
    right = biometricClamp(right, left + 1, video.videoWidth - 1);
    bottom = biometricClamp(bottom, top + 1, video.videoHeight - 1);

    const cropCanvas = makeBiometricPadTensor.canvas || (makeBiometricPadTensor.canvas = document.createElement("canvas"));
    cropCanvas.width = 80;
    cropCanvas.height = 80;
    const ctx = cropCanvas.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, 80, 80);
    ctx.drawImage(video, left, top, right - left, bottom - top, 0, 0, 80, 80);
    const pixels = ctx.getImageData(0, 0, 80, 80).data;
    const plane = 80 * 80;
    const input = new Float32Array(3 * plane);
    for (let i = 0, p = 0; i < plane; i++) {
        const r = pixels[p++];
        const g = pixels[p++];
        const b = pixels[p++];
        p++;
        input[i] = b;
        input[plane + i] = g;
        input[2 * plane + i] = r;
    }
    return new window.ort.Tensor("float32", input, [1, 3, 80, 80]);
}

function biometricSoftmax3(values) {
    const maxValue = Math.max.apply(null, values);
    const exps = values.map(function (value) { return Math.exp(value - maxValue); });
    const total = exps.reduce(function (sum, value) { return sum + value; }, 0) || 1;
    return exps.map(function (value) { return value / total; });
}

async function predictBiometricPadModel(session, cropScale, landmarks) {
    const result = await session.run({ [session.inputNames[0]]: makeBiometricPadTensor(cropScale, landmarks) });
    const logits = Array.from(result[session.outputNames[0]].data).slice(0, 3);
    if (logits.length !== 3) throw new Error("PAD model returned an unexpected output size.");
    const probabilities = biometricSoftmax3(logits);
    return probabilities[1]; // Reference prototype maps class index 1 to Real.
}

async function runBiometricPadDiagnostic() {
    if (biometricAiBusy) return;
    biometricAiBusy = true;
    updateBiometricAiButtons();
    try {
        const v2Scores = [];
        const v1Scores = [];
        for (let i = 0; i < BIOMETRIC_AI_MODEL_CONFIG.PAD_SAMPLE_COUNT; i++) {
            await biometricSleep(i === 0 ? 100 : 180);
            const landmarks = getCurrentBiometricLandmarks();
            setBiometricAiResult("PAD diagnostic · analyzing frame " + (i + 1) + " of 5…");
            v2Scores.push(await predictBiometricPadModel(biometricPadV2Session, 2.7, landmarks));
            v1Scores.push(await predictBiometricPadModel(biometricPadV1seSession, 4.0, landmarks));
        }
        const v2Median = biometricMedian(v2Scores);
        const v1Median = biometricMedian(v1Scores);
        const exploratoryFlag = v2Median >= BIOMETRIC_AI_MODEL_CONFIG.EXPLORATORY_PAD_THRESHOLD && v1Median >= BIOMETRIC_AI_MODEL_CONFIG.EXPLORATORY_PAD_THRESHOLD;
        setBiometricAiResult(
            "PAD scores (median live-class probability): V2 " + v2Median.toFixed(4) + " · V1SE " + v1Median.toFixed(4) + ". " +
            (exploratoryFlag ? "Both passed the prototype's exploratory 0.80 flag." : "One or both did not meet the prototype's exploratory 0.80 flag.") +
            " This is not proof of liveness and does not authorize attendance."
        );
    } catch (error) {
        setBiometricAiResult("PAD diagnostic failed: " + (error && error.message ? error.message : String(error)));
    } finally {
        biometricAiBusy = false;
        updateBiometricAiButtons();
    }
}

async function requestRegisteredTemplateComparisonTest(candidateEmbedding) {
    if (!app || !app.ticket || !app.parentOrigin || !attendanceParentWindow) {
        throw new Error("Open the camera from the authenticated RJAB Employee Portal before running the server comparison test.");
    }
    const requestId = (window.crypto && typeof window.crypto.randomUUID === "function")
        ? window.crypto.randomUUID().replace(/-/g, "")
        : (Date.now().toString(36) + Math.random().toString(36).slice(2, 18));
    const candidate = Array.from(candidateEmbedding, function (value) { return Number(value); });

    return await new Promise(function (resolve, reject) {
        const timeoutId = window.setTimeout(function () {
            if (pendingBiometricWorkerCompare && pendingBiometricWorkerCompare.requestId === requestId) {
                pendingBiometricWorkerCompare = null;
            }
            candidate.fill(0);
            reject(new Error("No response was received from the RJAB portal. Keep the portal tab open and try again with a new camera session."));
        }, 60000);

        pendingBiometricWorkerCompare = {
            requestId: requestId,
            resolve: resolve,
            reject: reject,
            timeoutId: timeoutId
        };

        try {
            attendanceParentWindow.postMessage({
                type: "RJAB_BIOMETRIC_COMPARE_TEMPLATE_TEST",
                version: 1,
                ticket: app.ticket,
                requestId: requestId,
                modelVersion: BIOMETRIC_AI_MODEL_CONFIG.MODEL_VERSION,
                dimension: candidate.length,
                embedding: candidate
            }, app.parentOrigin);
            // postMessage performs a structured clone synchronously.
            candidate.fill(0);
        } catch (error) {
            window.clearTimeout(timeoutId);
            pendingBiometricWorkerCompare = null;
            candidate.fill(0);
            reject(new Error("Unable to send the test vector to the RJAB portal: " + (error && error.message ? error.message : String(error))));
        }
    });
}

async function runRegisteredTemplateComparisonTest() {
    if (biometricAiBusy) return;
    biometricAiBusy = true;
    updateBiometricAiButtons();
    let candidate = null;
    try {
        setBiometricAiResult("Capturing 3 face samples for a test-only comparison with your encrypted pending registration…");
        candidate = await captureBiometricTestEmbeddingSet("Registered-template comparison");
        setBiometricAiResult("Sending the candidate vector through the authenticated RJAB portal. The browser will not receive the registered template…");
        const result = await requestRegisteredTemplateComparisonTest(candidate);
        const score = Number(result.similarity);
        const elapsed = Number(result.workerElapsedMs) || 0;
        if (!Number.isFinite(score)) throw new Error("The Worker returned an invalid similarity score.");
        setBiometricAiResult(
            "SERVER TEMPLATE COMPARISON · Similarity: " + score.toFixed(4) +
            ". Worker elapsed (wall-clock): " + elapsed.toFixed(2) + " ms. Record status: " +
            String(result.recordStatus || "PENDING_VALIDATION") +
            ". No acceptance threshold was applied. The registered template is still client-submitted/untrusted; this test cannot authorize attendance or payroll."
        );
    } catch (error) {
        setBiometricAiResult("Registered-template comparison failed: " + (error && error.message ? error.message : String(error)));
    } finally {
        if (candidate && typeof candidate.fill === "function") candidate.fill(0);
        biometricAiBusy = false;
        updateBiometricAiButtons();
    }
}

function bindBiometricAiTestControls() {
    if (!biometricAiTestMode) return;
    const referenceBtn = document.getElementById("biometricAiReferenceBtn");
    const compareBtn = document.getElementById("biometricAiCompareBtn");
    const workerCompareBtn = document.getElementById("biometricAiWorkerCompareBtn");
    const padBtn = document.getElementById("biometricAiPadBtn");
    const clearBtn = document.getElementById("biometricAiClearBtn");
    if (referenceBtn) referenceBtn.addEventListener("click", async function () {
        if (biometricAiBusy) return;
        biometricAiBusy = true;
        updateBiometricAiButtons();
        try {
            biometricReferenceEmbedding = await captureBiometricTestEmbeddingSet("Reference capture");
            setBiometricAiResult("Temporary test reference captured (512 values, 3-sample average). It exists only in this browser tab; it has not been saved or transmitted.");
        } catch (error) {
            biometricReferenceEmbedding = null;
            setBiometricAiResult("Reference capture failed: " + (error && error.message ? error.message : String(error)));
        } finally {
            biometricAiBusy = false;
            updateBiometricAiButtons();
        }
    });
    if (compareBtn) compareBtn.addEventListener("click", async function () {
        if (biometricAiBusy || !biometricReferenceEmbedding) return;
        biometricAiBusy = true;
        updateBiometricAiButtons();
        try {
            const candidate = await captureBiometricTestEmbeddingSet("Face comparison");
            const score = biometricCosineSimilarity(biometricReferenceEmbedding, candidate);
            setBiometricAiResult("Cosine similarity: " + score.toFixed(4) + " (range is typically -1 to 1). No accept/reject threshold is applied. Repeat with the same consenting tester, then a different consenting tester. This is a browser-only comparison against the temporary in-memory reference, not the employee's registered template.");
        } catch (error) {
            setBiometricAiResult("Face comparison failed: " + (error && error.message ? error.message : String(error)));
        } finally {
            biometricAiBusy = false;
            updateBiometricAiButtons();
        }
    });
    if (workerCompareBtn) workerCompareBtn.addEventListener("click", runRegisteredTemplateComparisonTest);
    if (padBtn) padBtn.addEventListener("click", runBiometricPadDiagnostic);
    if (clearBtn) clearBtn.addEventListener("click", function () {
        biometricReferenceEmbedding = null;
        setBiometricAiResult("Temporary test reference cleared from this page's JavaScript memory.");
        updateBiometricAiButtons();
    });
    updateBiometricAiButtons();
}

// =====================================
// DETECT FACES
// =====================================

function detectFaces() {

    if (!faceDetector) {

        throw new Error(
            "Face detector is not ready."
        );

    }

    const result = faceDetector.detect(canvas);

    const detections = result.detections || [];

    // No face
    if (detections.length === 0) {

        return {

            detected: false,

            faces: [],

            primaryFace: null

        };

    }

    // Find the largest detected face
    let primaryFace = detections[0];

    let primaryArea =
        primaryFace.boundingBox.width *
        primaryFace.boundingBox.height;

    for (let i = 1; i < detections.length; i++) {

        const face = detections[i];

        const area =
            face.boundingBox.width *
            face.boundingBox.height;

        if (area > primaryArea) {

            primaryFace = face;

            primaryArea = area;

        }

    }

    return {

        detected: true,

        faces: detections,

        primaryFace: primaryFace

    };

}

// =====================================
// LIVE FACE TRACKING
// =====================================

// One independent guide is maintained for every face returned by MediaPipe.
// The original #guide is reused for face 1; extra guides are cloned as needed.
let faceGuides = [];

function ensureFaceGuideCount(count) {

    while (faceGuides.length < count) {

        const clone = guide.cloneNode(true);
        clone.removeAttribute("id");
        clone.classList.add("face-guide-overlay");
        clone.hidden = true;
        clone.classList.remove("is-tracking");
        clone.style.left = "50%";
        clone.style.top = "50%";
        clone.style.width = "190px";
        clone.style.height = "240px";

        // The companion CSS gives .face-guide-overlay the same styling as #guide.
        guide.parentElement.appendChild(clone);
        faceGuides.push(clone);

    }

}

function hideUnusedFaceGuides(firstUnusedIndex = 0) {

    for (let i = firstUnusedIndex; i < faceGuides.length; i++) {
        faceGuides[i].classList.remove("is-tracking");
        faceGuides[i].hidden = true;
    }

}

function startLiveFaceTracking() {

    if (liveTrackingStarted) return;

    liveTrackingStarted = true;

    function trackingFrame() {

        if (
            liveFaceDetectorReady &&
            liveFaceDetector &&
            video &&
            !video.hidden &&
            guide &&
            video.readyState >= 2 &&
            video.videoWidth > 0 &&
            video.videoHeight > 0
        ) {

            const now = performance.now();

            // Around 8 detections per second is enough for smooth tracking
            // while keeping CPU usage reasonable on mobile devices.
            if (now - lastLiveDetectionAt >= 120) {

                lastLiveDetectionAt = now;

                try {

                    const result = liveFaceDetector.detectForVideo(video, now);
                    const detections = result.detections || [];

                    if (detections.length) {

                        noFaceFrames = 0;
                        ensureFaceGuideCount(detections.length);

                        // Draw a separate guide around EVERY detected face.
                        detections.forEach((detection, index) => {
                            moveGuideToFace(
                                detection.boundingBox,
                                faceGuides[index]
                            );
                        });

                        // Hide old guides if fewer faces are detected this frame.
                        hideUnusedFaceGuides(detections.length);

                    } else {

                        noFaceFrames++;

                        // Hide guides after a few missed frames to avoid flicker.
                        if (noFaceFrames >= 3) {
                            resetGuideToSearching();
                        }

                    }

                } catch (trackingError) {

                    console.warn("Live face tracking frame failed:", trackingError);

                }

            }

        }

        requestAnimationFrame(trackingFrame);

    }

    requestAnimationFrame(trackingFrame);

}

function moveGuideToFace(box, targetGuide = guide) {

    if (!box || !targetGuide || !video) return;

    // Only show this guide after a face has actually been detected.
    targetGuide.hidden = false;

    const displayWidth = video.clientWidth;
    const displayHeight = video.clientHeight;
    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;

    if (!displayWidth || !displayHeight || !sourceWidth || !sourceHeight) return;

    // Match the video's object-fit: cover crop so the guide aligns with
    // the face as it appears inside the visible square camera preview.
    const scale = Math.max(
        displayWidth / sourceWidth,
        displayHeight / sourceHeight
    );

    const renderedWidth = sourceWidth * scale;
    const renderedHeight = sourceHeight * scale;
    const cropX = (renderedWidth - displayWidth) / 2;
    const cropY = (renderedHeight - displayHeight) / 2;

    let faceLeft = (box.originX * scale) - cropX;
    const faceTop = (box.originY * scale) - cropY;
    const faceWidth = box.width * scale;
    const faceHeight = box.height * scale;

    // If CSS mirrors the video horizontally, mirror the detected X coordinate
    // for the visible preview so the guide stays on the same displayed side.
    const videoTransform = window.getComputedStyle(video).transform;
    if (videoTransform && videoTransform !== "none") {
        try {
            const matrix = new DOMMatrixReadOnly(videoTransform);
            if (matrix.a < 0) faceLeft = displayWidth - faceLeft - faceWidth;
        } catch (_) { /* Keep the normal coordinate mapping if unsupported. */ }
    }


    // Keep the box narrower and shorter, with more room above the hair.
    const guideWidth = Math.min(
        displayWidth * 0.94,
        Math.max(110, faceWidth * 1.55)
    );

    const guideHeight = Math.min(
        displayHeight * 0.94,
        Math.max(140, faceHeight * 1.55)
    );

    const centerX = Math.max(
        guideWidth / 2,
        Math.min(displayWidth - guideWidth / 2, faceLeft + faceWidth / 2)
    );

    // Shift the box slightly upward to add headroom and reduce space below.
    const desiredCenterY =
        faceTop + faceHeight / 2 - faceHeight * 0.20;

    const centerY = Math.max(
        guideHeight / 2,
        Math.min(displayHeight - guideHeight / 2, desiredCenterY)
    );


    targetGuide.style.left = `${video.offsetLeft + centerX}px`;
    targetGuide.style.top = `${video.offsetTop + centerY}px`;
    targetGuide.style.width = `${guideWidth}px`;
    targetGuide.style.height = `${guideHeight}px`;
    targetGuide.classList.add("is-tracking");

}

function resetGuideToSearching() {

    if (!guide) return;

    faceGuides.forEach((faceGuide) => {
        faceGuide.classList.remove("is-tracking");
        faceGuide.hidden = true;
        faceGuide.style.left = "50%";
        faceGuide.style.top = "50%";
        faceGuide.style.width = "190px";
        faceGuide.style.height = "240px";
    });

}

// =====================================
// FACE SIZE TEST
// =====================================

function checkFaceSize(face) {

    if (!face || !face.boundingBox) {

        return {
            ok: false,
            message: "No face"
        };

    }

    const box = face.boundingBox;

    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;

    const faceWidthRatio = box.width / canvasWidth;
    const faceHeightRatio = box.height / canvasHeight;

    // Face should not be too small
    if (
        faceWidthRatio < 0.20 ||
        faceHeightRatio < 0.20
    ) {

        return {
            ok: false,
            message: "Face too small"
        };

    }

    // Face should not be too large
    if (
        faceWidthRatio > 0.80 ||
        faceHeightRatio > 0.80
    ) {

        return {
            ok: false,
            message: "Face too large"
        };

    }

    return {
        ok: true,
        message: "Face size OK"
    };

}


// =====================================
// FACE POSITION TEST
// =====================================

function checkFacePosition(face) {

    if (!face || !face.boundingBox) {

        return {
            ok: false,
            message: "No face"
        };

    }

    const box = face.boundingBox;

    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;

    // Center point of detected face
    const faceCenterX =
        box.originX + (box.width / 2);

    const faceCenterY =
        box.originY + (box.height / 2);

    const canvasCenterX =
        canvasWidth / 2;

    const canvasCenterY =
        canvasHeight / 2;

    // Allowed distance from center
    const allowedX =
        canvasWidth * 0.25;

    const allowedY =
        canvasHeight * 0.25;

    const distanceX =
        Math.abs(faceCenterX - canvasCenterX);

    const distanceY =
        Math.abs(faceCenterY - canvasCenterY);

    if (
        distanceX > allowedX ||
        distanceY > allowedY
    ) {

        return {
            ok: false,
            message: "Face not centered"
        };

    }

    return {
        ok: true,
        message: "Face position OK"
    };

}

const video = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const preview = document.getElementById("preview");

const captureBtn = document.getElementById("captureBtn");
const retakeBtn = document.getElementById("retakeBtn");
const useBtn = document.getElementById("useBtn");

captureBtn.disabled = true;
captureBtn.hidden = true;

const controls = document.querySelector(".buttons");

// =====================================
// NATURAL CAMERA FILTERS
// Filters are applied to the live preview and baked into the captured image.
// They adjust light/color only; they do not reshape facial features.
// =====================================
const cameraFilters = {
    original: { label: "Original", css: "none" },
    enhance:  { label: "Enhance",  css: "brightness(1.06) contrast(1.08) saturate(1.04)" },
    bright:   { label: "Bright",   css: "brightness(1.12) contrast(1.03)" },
    natural:  { label: "Natural",  css: "brightness(1.03) saturate(1.06)" },
    vivid:    { label: "Vivid",    css: "saturate(1.22) contrast(1.08) brightness(1.03)" }
};


let activeCameraFilter = "enhance";


let nightModeEnabled = false;
let screenFlashEnabled = false;

const nightModeBtn = document.getElementById("nightModeBtn");
const screenFlashBtn = document.getElementById("screenFlashBtn");

// Combine the existing filter with Night Mode when enabled.
function getActiveCameraFilterCss() {
    const baseFilter =
        cameraFilters[activeCameraFilter]?.css || "none";

    const filters = [];

    if (baseFilter !== "none") {
        filters.push(baseFilter);
    }

    if (nightModeEnabled) {
        filters.push("brightness(1.55) contrast(0.92) saturate(1.08)");
    }

    return filters.join(" ") || "none";
}

function updateCameraPreviewFilter() {
    const filterCss = getActiveCameraFilterCss();

    video.style.filter = filterCss;
    video.style.webkitFilter = filterCss;
}

// NIGHT MODE ON/OFF
if (nightModeBtn) {
    nightModeBtn.addEventListener("click", function () {
        nightModeEnabled = !nightModeEnabled;

        nightModeBtn.classList.toggle(
            "is-active",
            nightModeEnabled
        );

        nightModeBtn.textContent =
            nightModeEnabled ? "☾ NIGHT ON" : "☾ NIGHT OFF";

        nightModeBtn.setAttribute(
            "aria-pressed",
            String(nightModeEnabled)
        );

        updateCameraPreviewFilter();
    });
}

// SCREEN FLASH ON/OFF
if (screenFlashBtn) {
    screenFlashBtn.addEventListener("click", function () {
        screenFlashEnabled = !screenFlashEnabled;

        screenFlashBtn.classList.toggle(
            "is-active",
            screenFlashEnabled
        );

        screenFlashBtn.textContent =
            screenFlashEnabled ? "⚡ FLASH ON" : "⚡ FLASH OFF";

        screenFlashBtn.setAttribute(
            "aria-pressed",
            String(screenFlashEnabled)
        );
    });
}


function applyCameraFilter(filterKey) {
    if (!cameraFilters[filterKey]) return;

    activeCameraFilter = filterKey;
    updateCameraPreviewFilter();

    document.querySelectorAll(".camera-filter-btn").forEach((button) => {
        const selected = button.dataset.filter === filterKey;
        button.classList.toggle("active", selected);
        button.setAttribute("aria-pressed", selected ? "true" : "false");
    });
}

function installCameraFilterControls() {
    if (document.getElementById("cameraFilterBar") || !controls || !controls.parentElement) return;

    const style = document.createElement("style");
    style.id = "cameraFilterStyles";
    style.textContent = `
        #cameraFilterBar {
            display:flex; gap:8px; align-items:center; justify-content:flex-start;
            width:100%; max-width:100%; box-sizing:border-box; overflow-x:auto;
            padding:10px 2px 12px; margin:8px 0 4px;
            scrollbar-width:none; -webkit-overflow-scrolling:touch;
        }
        #cameraFilterBar::-webkit-scrollbar { display:none; }
        .camera-filter-btn {
            flex:0 0 auto; min-width:72px; padding:9px 13px;
            border:1px solid rgba(255,255,255,.34); border-radius:999px;
            background:rgba(255,255,255,.10); color:#fff;
            -webkit-backdrop-filter:blur(14px) saturate(150%);
            backdrop-filter:blur(14px) saturate(150%);
            box-shadow:inset 0 1px 1px rgba(255,255,255,.16), 0 3px 10px rgba(0,0,0,.12);
            font:600 13px/1.2 system-ui,-apple-system,sans-serif;
            white-space:nowrap; cursor:pointer; touch-action:manipulation;
            transition:background .18s ease,border-color .18s ease,transform .18s ease;
        }
        .camera-filter-btn.active {
            background:rgba(255,255,255,.27); border-color:rgba(255,255,255,.82);
            box-shadow:inset 0 1px 2px rgba(255,255,255,.35),0 0 0 1px rgba(255,255,255,.10);
        }
        .camera-filter-btn:active { transform:scale(.97); }
        @media (max-width:480px) {
            #cameraFilterBar { gap:7px; }
            .camera-filter-btn { padding:9px 12px; min-width:68px; font-size:12px; }
        }
    `;
    document.head.appendChild(style);

    const bar = document.createElement("div");
    bar.id = "cameraFilterBar";
    bar.setAttribute("aria-label", "Camera filters");
    
    Object.entries(cameraFilters).forEach(([key, filter]) => {
        const button = document.createElement("button");
    
        button.type = "button";
        button.className = "camera-filter-btn";
        button.dataset.filter = key;
        button.setAttribute("aria-pressed", "false");
    
        // Center the filter label consistently
        button.textContent = filter.label;
        button.style.display = "inline-flex";
        button.style.alignItems = "center";
        button.style.justifyContent = "center";
        button.style.textAlign = "center";
        button.style.whiteSpace = "nowrap";
        button.style.boxSizing = "border-box";
        button.style.flexShrink = "0";
        button.style.lineHeight = "1";
    
        button.addEventListener("click", () => {
            applyCameraFilter(key);
        });
    
        bar.appendChild(button);
    });
    
    controls.parentElement.insertBefore(bar, controls);
    applyCameraFilter(activeCameraFilter);
}

installCameraFilterControls();

const guide = document.getElementById("guide");
faceGuides = [guide];
// Keep the face guide hidden until live detection finds a face.
guide.hidden = true;
const status = document.getElementById("status");
const qualityStatus = document.getElementById("qualityStatus");
let audioContext = null;
let captureLocked = false;
let cameraRetry = 0;

// Prevent duplicate verification submissions from the same Camera page.
let uploadInProgress = false;

let loadingInterval = null;

// =====================================
// APPLICATION STATE
// =====================================

const app = {
    stream: null,
    captured: false,
    photoData: "",
    latitude: "",
    longitude: "",
    accuracy: "",
    address: "",
    gpsTimestamp: "",
    clientCapturedAt: "",
    ticket: "",
    parentOrigin: "",
    type: "",
    cameraVersion: "attendance-camera-1.1.0-phase2g1"
};

// Bind the optional test controls only after the attendance state exists.
bindBiometricAiTestControls();

// ============================
// SECURE ATTENDANCE HANDOFF
// ============================
const launchParams = new URLSearchParams(window.location.search);
app.ticket = launchParams.get("ticket") || "";
app.parentOrigin = launchParams.get("parentOrigin") || "";
const requestedMode = String(launchParams.get("mode") || "").trim().toUpperCase();
app.type = requestedMode === "TIMEIN" ? "TIME IN" : requestedMode === "TIMEOUT" ? "TIME OUT" : "";

// Remove ticket and launch parameters from the visible URL immediately.
try {
    window.history.replaceState({}, document.title, window.location.pathname);
} catch (historyError) {
    console.warn("Unable to clean camera URL:", historyError);
}

let attendanceParentWindow = null;
let pendingAttendanceAck = null;
let pendingBiometricWorkerCompare = null;

function validateAttendanceLaunch() {
    if (!app.ticket || app.ticket.length < 24 || app.ticket.length > 512) {
        throw new Error("Attendance launch ticket is missing or invalid. Please reopen the camera from the RJAB Employee Portal.");
    }
    if (!app.type) {
        throw new Error("Invalid attendance action. Please start again using TIME IN or TIME OUT in the RJAB Employee Portal.");
    }
    let parsedParent;
    try { parsedParent = new URL(app.parentOrigin); }
    catch (error) { throw new Error("The attendance portal origin is missing or invalid. Please reopen the camera from the RJAB Employee Portal."); }
    if (parsedParent.protocol !== "https:" || parsedParent.origin !== app.parentOrigin) {
        throw new Error("The attendance portal must use a valid HTTPS origin.");
    }
    if (window.opener && !window.opener.closed) attendanceParentWindow = window.opener;
    else if (window.parent && window.parent !== window) attendanceParentWindow = window.parent;
    else throw new Error("Please open this camera from the RJAB Employee Portal. Direct access is not allowed.");

    const context = document.getElementById("attendanceContext");
    if (context) context.textContent = "RJAB SECURE " + app.type + " CAPTURE";
    window.addEventListener("message", handleAttendancePortalMessage);
    return true;
}

function handleAttendancePortalMessage(event) {
    if (!attendanceParentWindow || event.source !== attendanceParentWindow) return;
    if (event.origin !== app.parentOrigin) return;
    const data = event.data || {};
    if (data.ticket !== app.ticket) return;

    if (data.type === "RJAB_BIOMETRIC_COMPARE_TEMPLATE_TEST_RESULT" && Number(data.version) === 1) {
        if (!pendingBiometricWorkerCompare || data.requestId !== pendingBiometricWorkerCompare.requestId) return;
        const pendingCompare = pendingBiometricWorkerCompare;
        pendingBiometricWorkerCompare = null;
        clearTimeout(pendingCompare.timeoutId);
        if (data.ok === true) pendingCompare.resolve(data);
        else pendingCompare.reject(new Error(data.message || "The RJAB portal could not complete the test comparison."));
        return;
    }

    if (data.type !== "RJAB_ATTENDANCE_CAPTURE_ACK" || !pendingAttendanceAck) return;
    const pending = pendingAttendanceAck;
    pendingAttendanceAck = null;
    clearTimeout(pending.timeoutId);
    if (data.ok === true) pending.resolve(data);
    else pending.reject(new Error(data.message || "The attendance portal could not save this capture."));
}

function startLoadingAnimation() {

    stopLoadingAnimation();

    const frames = [

        "📷 Initializing Camera.",
        "📷 Initializing Camera..",
        "📷 Initializing Camera..."

    ];

    let i = 0;

    status.innerHTML = frames[0];

    loadingInterval = setInterval(function () {

        i = (i + 1) % frames.length;

        status.innerHTML = frames[i];

    }, 400);

}

function stopLoadingAnimation() {

    clearInterval(loadingInterval);

    loadingInterval = null;

}
// ============================
// PHOTO QUALITY STATUS
// ============================

function showQualityChecking() {

    qualityStatus.style.display = "block";

    qualityStatus.innerHTML = `
        <div style="
            font-size:17px;
            font-weight:700;
            margin-bottom:10px;
        ">
            🔍 AI Photo Quality Scan
        </div>

        <div style="
            font-size:14px;
            opacity:.85;
        ">
            Checking image quality...
        </div>
    `;

}
// ============================
// QUALITY PASSED
// ============================

function showQualityPassed() {

    qualityStatus.style.display = "block";

    qualityStatus.innerHTML = `
        <div style="
            font-size:17px;
            font-weight:700;
            color:#22c55e;
            margin-bottom:10px;
        ">
            ✅ Photo Quality Passed
        </div>

        <div style="
            font-size:14px;
            opacity:.85;
        ">
            Your photo is ready for verification.
        </div>
    `;

}

// ============================
// QUALITY FAILED
// ============================

function showQualityFailed(reason) {

    qualityStatus.style.display = "block";

    qualityStatus.innerHTML = `
        <div style="
            font-size:17px;
            font-weight:700;
            color:#ef4444;
            margin-bottom:10px;
        ">
            ❌ Photo Quality Failed
        </div>

        <div style="
            font-size:14px;
            opacity:.9;
        ">
            ${reason}
        </div>
    `;

}
// ============================
// CHECK BRIGHTNESS
// ============================

function checkBrightness() {

    const ctx = canvas.getContext("2d");

    const image = ctx.getImageData(
        0,
        0,
        canvas.width,
        canvas.height
    );

    const data = image.data;

    let total = 0;

    for (let i = 0; i < data.length; i += 4) {

        total +=
            (data[i] +
             data[i + 1] +
             data[i + 2]) / 3;

    }

    const brightness =
        total / (data.length / 4);

    return brightness;

}
// ============================
// CHECK BLUR V2
// ============================

function checkBlur() {

const sampleSize = 160;

const tempCanvas = document.createElement("canvas");

tempCanvas.width = sampleSize;
tempCanvas.height = sampleSize;

const tctx = tempCanvas.getContext("2d");

const source = canvas;

// ----- CENTER CROP -----

const cropSize = Math.min(
    source.width,
    source.height
) * 0.60;

const sx = (source.width - cropSize) / 2;

const sy = (source.height - cropSize) / 2;

tctx.drawImage(

    source,

    sx,
    sy,

    cropSize,
    cropSize,

    0,
    0,

    sampleSize,
    sampleSize

);

    const img = tctx.getImageData(
        0,
        0,
        sampleSize,
        sampleSize
    );

    const data = img.data;

    let sum = 0;

    let count = 0;

    for (let y = 1; y < sampleSize - 1; y++) {

        for (let x = 1; x < sampleSize - 1; x++) {

            const center =
                ((y * sampleSize) + x) * 4;

            const left =
                center - 4;

            const right =
                center + 4;

            const top =
                center - sampleSize * 4;

            const bottom =
                center + sampleSize * 4;

            const c =
                (data[center] +
                 data[center+1] +
                 data[center+2]) / 3;

            const l =
                (data[left] +
                 data[left+1] +
                 data[left+2]) / 3;

            const r =
                (data[right] +
                 data[right+1] +
                 data[right+2]) / 3;

            const t =
                (data[top] +
                 data[top+1] +
                 data[top+2]) / 3;

            const b =
                (data[bottom] +
                 data[bottom+1] +
                 data[bottom+2]) / 3;

            const lap =
                (4 * c) - l - r - t - b;

            sum += Math.abs(lap);

            count++;

        }

    }

    return sum / count;

}
// ============================
// CHECK FACE BLUR
// ============================

function checkFaceBlur(face) {

    if (!face || !face.boundingBox) {

        return null;

    }

    const box = face.boundingBox;

    const source = canvas;

    // Add padding around the detected face
    const paddingX = box.width * 0.20;
    const paddingY = box.height * 0.20;

    let sx = box.originX - paddingX;
    let sy = box.originY - paddingY;

    let sw = box.width + (paddingX * 2);
    let sh = box.height + (paddingY * 2);

    // Keep crop inside canvas
    if (sx < 0) {
        sw += sx;
        sx = 0;
    }

    if (sy < 0) {
        sh += sy;
        sy = 0;
    }

    if (sx + sw > source.width) {
        sw = source.width - sx;
    }

    if (sy + sh > source.height) {
        sh = source.height - sy;
    }

    const sampleSize = 160;

    const tempCanvas =
        document.createElement("canvas");

    tempCanvas.width = sampleSize;
    tempCanvas.height = sampleSize;

    const tctx =
        tempCanvas.getContext("2d");

    tctx.drawImage(

        source,

        sx,
        sy,
        sw,
        sh,

        0,
        0,
        sampleSize,
        sampleSize

    );

    const img = tctx.getImageData(
        0,
        0,
        sampleSize,
        sampleSize
    );

    const data = img.data;

    let sum = 0;
    let count = 0;

    for (
        let y = 1;
        y < sampleSize - 1;
        y++
    ) {

        for (
            let x = 1;
            x < sampleSize - 1;
            x++
        ) {

            const center =
                ((y * sampleSize) + x) * 4;

            const left =
                center - 4;

            const right =
                center + 4;

            const top =
                center - sampleSize * 4;

            const bottom =
                center + sampleSize * 4;

            const c =
                (
                    data[center] +
                    data[center + 1] +
                    data[center + 2]
                ) / 3;

            const l =
                (
                    data[left] +
                    data[left + 1] +
                    data[left + 2]
                ) / 3;

            const r =
                (
                    data[right] +
                    data[right + 1] +
                    data[right + 2]
                ) / 3;

            const t =
                (
                    data[top] +
                    data[top + 1] +
                    data[top + 2]
                ) / 3;

            const b =
                (
                    data[bottom] +
                    data[bottom + 1] +
                    data[bottom + 2]
                ) / 3;

            const lap =
                (4 * c) - l - r - t - b;

            sum += Math.abs(lap);

            count++;

        }

    }

    return sum / count;

}
// ============================
// ANALYZE PHOTO QUALITY
// ============================

function analyzePhotoQuality() {

    app.scannerSkipped = false;

    if (!faceDetectorReady || !faceDetector) {

        return {

            pass: false,

            reason:
                "🤖 AI Scanner is still loading.<br><br>" +
                "Please wait a moment and retake your photo."

        };

    }

    // ----------------------------
    // Brightness Check
    // ----------------------------

    const brightness = checkBrightness();

    if (brightness < 35) {

        return {

            pass: false,

            reason:
                "💡 Lighting is too dark.<br><br>Please retake your photo."

        };

    }

    // ----------------------------
    // FACE DETECTION
    // ----------------------------

    let faceResult;

try {

    faceResult = detectFaces();

} catch (err) {

    const msg = (err && err.message) || String(err);

    // WebGL/MediaPipe error sa device na ito: brightness + blur lang
    if (/GLctx|WebGL|texture/i.test(msg)) {

        app.scannerSkipped = true;

        if (checkBlur() < 5) {
            return {
                pass: false,
                reason:
                    "📷 Image is blurry.<br><br>Please hold the camera steady and retake your photo."
            };
        }

        return { pass: true, scannerSkipped: true };
    }

    throw err;
}

if (!faceResult.primaryFace) {

        return {

            pass: false,

            reason:
                "🙂 No face detected.<br><br>Please make sure your face is clearly visible."

        };

    }

    // ----------------------------
    // FACE BLUR CHECK
    // ----------------------------

    const faceBlur =
        checkFaceBlur(faceResult.primaryFace);

    if (faceBlur === null) {

        return {

            pass: false,

            reason:
                "🙂 Unable to verify face quality.<br><br>Please retake your photo."

        };

    }

    // Final threshold
    if (faceBlur < 5) {

        return {

            pass: false,

            reason:
                "📷 Image is blurry.<br><br>Please hold the camera steady and retake your photo."

        };

    }

    // ----------------------------
    // PASSED
    // ----------------------------

    return {

        pass: true,

        brightness: brightness,

        blur: faceBlur,

        facesDetected: faceResult.faces.length

    };

}
// ============================
// START CAMERA
// ============================

async function startCamera() {

    startLoadingAnimation();

    const timeout = setTimeout(function () {

        stopLoadingAnimation();

        if (app.stream) {

            app.stream.getTracks().forEach(track => track.stop());

            app.stream = null;

        }

        status.innerHTML =
        "❌ Camera initialization timed out.<br>Please reopen the camera.";

    },10000);

    try {

        if (app.stream) {
            app.stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            app.stream = null;
        }

        app.stream = await navigator.mediaDevices.getUserMedia({

            video: {
                facingMode: "user"
            },

            audio: false

        });

        video.srcObject = app.stream;

        video.autoplay = true;
        video.muted = true;
        video.playsInline = true;

        video.setAttribute("playsinline", "");
        video.setAttribute("webkit-playsinline", "");

        video.onloadedmetadata = async function () {

            try {

                await video.play();

            } catch (err) {

                console.log(err);

            }

        };

        clearTimeout(timeout);

        stopLoadingAnimation();

        cameraRetry = 0;

        // Reset camera state
        captureLocked = false;
        
        cameraReady = true;

        startLiveFaceTracking();
        
        retakeBtn.disabled = false;
        
        setVerifyButton(true);
        
        updateCameraReadyState();
        updateBiometricAiButtons();
        
        if (!faceDetectorReady) {
        
            status.innerHTML = "🤖 Initializing AI Scanner...";
        
        }

    }

    catch (error) {

        console.error(error);

        clearTimeout(timeout);

        stopLoadingAnimation();

        if (cameraRetry < 1) {

            cameraRetry++;

            status.innerHTML = "🔄 Retrying Camera...";

            setTimeout(startCamera, 1000);

            return;

        }

        status.innerHTML =
        "❌ Unable to access camera.<br>Please reopen the camera.";

    }

}
function playShutterSound() {

    try {

        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }

        const now = audioContext.currentTime;

        // Main click
        const osc = audioContext.createOscillator();
        const gain = audioContext.createGain();

        osc.type = "triangle";
        osc.frequency.setValueAtTime(1400, now);
        osc.frequency.exponentialRampToValueAtTime(500, now + 0.05);

        gain.gain.setValueAtTime(0.22, now);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.06);

        osc.connect(gain);
        gain.connect(audioContext.destination);

        osc.start(now);
        osc.stop(now + 0.06);

        // Mechanical click
        const osc2 = audioContext.createOscillator();
        const gain2 = audioContext.createGain();

        osc2.type = "square";
        osc2.frequency.setValueAtTime(260, now + 0.01);

        gain2.gain.setValueAtTime(0.08, now + 0.01);
        gain2.gain.exponentialRampToValueAtTime(0.0001, now + 0.05);

        osc2.connect(gain2);
        gain2.connect(audioContext.destination);

        osc2.start(now + 0.01);
        osc2.stop(now + 0.05);

    } catch (err) {

        console.log(err);

    }

}
// ============================
// CAPTURE
// ============================

let captureEventHandled = false;

function handleCaptureButton(event) {

    if (event) {
        event.preventDefault();
    }

    if (captureEventHandled) {
        return;
    }

    if (captureLocked) {
        return;
    }

    if (captureBtn.disabled) {
        return;
    }

    captureEventHandled = true;
    captureLocked = true;

    startCountdown();

    setTimeout(function () {

        captureEventHandled = false;

    }, 500);

}


// iOS / Android browser pointer input
captureBtn.addEventListener(
    "pointerup",
    handleCaptureButton
);


// Touch fallback
captureBtn.addEventListener(
    "touchend",
    handleCaptureButton,
    {
        passive: false
    }
);


async function startCountdown() {

    if (!video.videoWidth || !video.videoHeight) {
        captureLocked = false;
        captureBtn.disabled = false;

        alert("Camera is not ready.");
        return;
    }

    captureBtn.disabled = true;

    const countdown = document.getElementById("countdown");

    if (!countdown) {
        console.error("Countdown element not found.");
        captureLocked = false;
        captureBtn.disabled = false;
        return;
    }

    countdown.hidden = false;
    status.innerHTML = "📸 Digital Scanner — Get Ready...";

    for (let i = 3; i >= 1; i--) {

        countdown.textContent = i;

        // Restart the number animation on every count
        countdown.classList.remove("scanner-number");
        void countdown.offsetWidth;
        countdown.classList.add("scanner-number");

        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    countdown.hidden = true;
    countdown.classList.remove("scanner-number");

    if (navigator.vibrate) {
        navigator.vibrate(40);
    }

    playShutterSound();

    
const flash = document.getElementById("flash");

if (screenFlashEnabled && flash) {

    // Restart the flash animation on every capture.
    flash.classList.remove("active");
    void flash.offsetWidth;
    flash.classList.add("active");

    // Give the screen flash a brief moment before capture.
    setTimeout(function() {
        capturePhoto();
    }, 220);
    
    // Remove the flash after the animation has time to finish.
    setTimeout(function() {
        flash.classList.remove("active");
    }, 650);
    
    
    } else {

    // No screen flash when the option is OFF.
    setTimeout(function() {
        capturePhoto();
    }, 30);

}

}

function capturePhoto(){

    if (!faceDetectorReady || !faceDetector) {

        console.warn(
            "⚠️ Capture blocked: MediaPipe is still loading."
        );

        status.style.display = "block";

        status.innerHTML =
            "🤖 AI Scanner is still loading...<br><br>" +
            "Please wait a moment before taking your photo.";

        return;

    }

    if (!video.videoWidth || !video.videoHeight) {

        captureLocked = false;

        captureBtn.disabled = false;

        alert("Camera is not ready.");

        return;

    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");

    // Bake the selected natural light/color filter into the saved photo.
    // Canvas filtering changes pixels only; no facial geometry is altered.
    ctx.filter = getActiveCameraFilterCss();

    // I-mirror ang captured photo para tumugma sa live preview ng front camera
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);
    ctx.restore();

    ctx.filter = "none";

    app.photoData = canvas.toDataURL("image/jpeg", 0.95);
    app.clientCapturedAt = new Date().toISOString();

    if (app.stream) {
        app.stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        app.stream = null;
    }
    updateBiometricAiButtons();

    preview.src = app.photoData;

    preview.hidden = false;
    video.hidden = true;
    hideUnusedFaceGuides(0);

    captureBtn.hidden = true;
    captureBtn.disabled = true;

    retakeBtn.hidden = false;
    useBtn.hidden = false;

    app.captured = true;

// =====================================
// FACE DETECTION TEST
// =====================================

try {

    const faceResult = detectFaces();

console.log(
    "🙂 Faces detected:",
    faceResult.faces.length
);

if (faceResult.primaryFace) {

    const box = faceResult.primaryFace.boundingBox;

    console.log(
        "🎯 Primary face detected:",
        {
            x: Math.round(box.originX),
            y: Math.round(box.originY),
            width: Math.round(box.width),
            height: Math.round(box.height)
        }
    );


    // ================================
    // FACE SIZE TEST
    // ================================

    const sizeResult =
        checkFaceSize(faceResult.primaryFace);

    console.log(
        sizeResult.ok
            ? "📏 Face Size: OK"
            : "📏 Face Size: " + sizeResult.message
    );


    // ================================
    // FACE POSITION TEST
    // ================================

    const positionResult =
        checkFacePosition(faceResult.primaryFace);

    console.log(
        positionResult.ok
            ? "📍 Face Position: OK"
            : "📍 Face Position: " + positionResult.message
    );

// ================================
// FACE BLUR TEST
// ================================

const fullImageBlur = checkBlur();

const faceBlur =
    checkFaceBlur(faceResult.primaryFace);

console.log(
    "📊 Full Image Blur:",
    fullImageBlur.toFixed(2)
);

console.log(
    "🙂 Face Blur:",
    faceBlur !== null
        ? faceBlur.toFixed(2)
        : "N/A"
);


} else {

    console.log(
        "❌ No face detected"
    );

}
}
catch (error) {

    console.error(
        "❌ Face detection test failed:",
        error
    );

}

// Hide status panel habang preview
status.style.display = "none";

showQualityChecking();

// Disable muna habang nagsa-scan
setVerifyButton(false);

// Simulate AI Scan
setTimeout(function () {

    const result = analyzePhotoQuality();

    if (!result.pass) {

        showQualityFailed(result.reason);

        setVerifyButton(false);

    } else {

showQualityPassed();

setVerifyButton(true);

    }

}, 700);

}
// ============================
// RETAKE
// ============================

retakeBtn.onclick = async function () {

    preview.hidden = true;
    video.hidden = false;
    noFaceFrames = 0;
    resetGuideToSearching();

    captureBtn.hidden = false;
    captureBtn.disabled = false;

    captureLocked = false;

    retakeBtn.hidden = true;
    useBtn.hidden = true;

    app.captured = false;
    app.photoData = "";

    preview.src = "";
    applyCameraFilter(activeCameraFilter);

    app.latitude = "";
    app.longitude = "";
    app.accuracy = "";
    app.address = "";
    app.gpsTimestamp = "";
    app.clientCapturedAt = "";
    qualityStatus.style.display = "none";

    status.style.display = "block";

    setVerifyButton(true);

    await startCamera();

};

// ============================
// GET GPS
// ============================

function getGPS() {

    return new Promise(function(resolve, reject){

        if (!navigator.geolocation){

            reject("GPS not supported");
            return;

        }

        navigator.geolocation.getCurrentPosition(

            function(position){

                app.latitude  = position.coords.latitude.toFixed(6);
                app.longitude = position.coords.longitude.toFixed(6);
                app.accuracy  = Math.round(position.coords.accuracy);
                app.gpsTimestamp = position.timestamp ? new Date(position.timestamp).toISOString() : new Date().toISOString();

                resolve();

            },

            function(error){

                reject(error.message);

            },

            {
                enableHighAccuracy: true,
                timeout: 15000,
                maximumAge: 0
            }

        );

    });

}

// ============================
// GET ADDRESS
// ============================

async function getAddress() {

    if (!app.latitude || !app.longitude) {

        app.address = "GPS not available";
        return;

    }

    try {

        const url =
            "https://nominatim.openstreetmap.org/reverse?format=jsonv2" +
            "&lat=" + app.latitude +
            "&lon=" + app.longitude;

        const response = await fetch(url, {

            headers: {
                "Accept": "application/json"
            }

        });

        const data = await response.json();

        app.address = data.display_name || "Unknown Address";

    }

    catch (error) {

        console.error(error);

        app.address = "Address not available";

    }

}

// ============================
// DRAW MULTILINE TEXT
// ============================

function drawWrappedText(ctx, text, x, y, maxWidth, lineHeight) {

    const words = (text || "").split(" ");

    let line = "";

    for (let i = 0; i < words.length; i++) {

        const testLine = line + words[i] + " ";

        const width = ctx.measureText(testLine).width;

        if (width > maxWidth && i > 0) {

            ctx.fillText(line, x, y);

            line = words[i] + " ";

            y += lineHeight;

        } else {

            line = testLine;

        }

    }

    ctx.fillText(line, x, y);

    return y + lineHeight;

}
// =====================================
// DRAW LIMITED WRAPPED TEXT
// =====================================

function drawWrappedLimitedText(
    ctx,
    text,
    x,
    y,
    maxWidth,
    lineHeight,
    maxLines
) {

    const words = (text || "").split(" ");

    const lines = [];

    let line = "";

    for (let i = 0; i < words.length; i++) {

        const testLine = line + words[i] + " ";

        if (
            ctx.measureText(testLine).width > maxWidth &&
            line !== ""
        ) {

            lines.push(line.trim());

            line = words[i] + " ";

        } else {

            line = testLine;

        }

    }

    lines.push(line.trim());

    if (lines.length > maxLines) {

        lines.length = maxLines;

        while (
            ctx.measureText(lines[maxLines - 1] + "...").width > maxWidth
        ) {

            const arr = lines[maxLines - 1].split(" ");

            arr.pop();

            lines[maxLines - 1] = arr.join(" ");

        }

        lines[maxLines - 1] += "...";

    }

    for (let i = 0; i < lines.length; i++) {

        ctx.fillText(
            lines[i],
            x,
            y
        );

        y += lineHeight;

    }

    return lines.length;

}

// ============================
// COUNT WRAPPED LINES
// ============================

function getWrappedLineCount(ctx, text, maxWidth) {

    const words = (text || "").split(" ");

    let line = "";
    let count = 1;

    for (let i = 0; i < words.length; i++) {

        const test = line + words[i] + " ";

        if (ctx.measureText(test).width > maxWidth && i > 0) {

            count++;
            line = words[i] + " ";

        } else {

            line = test;

        }

    }

    return count;

}
// =====================================
// DRAW TWO COLUMN ROW
// =====================================

function drawInfoRow(
    ctx,
    leftIcon,
    leftText,
    rightIcon,
    rightText,
    y,
    leftX,
    rightX,
    fontSize
){

    ctx.font = `${fontSize}px Arial`;
    ctx.fillStyle = "#FFFFFF";
    ctx.textBaseline = "top";

    ctx.fillText(
        leftIcon + " " + leftText,
        leftX,
        y
    );

    ctx.fillText(
        rightIcon + " " + rightText,
        rightX,
        y
    );

    return y + (fontSize * 1.6);

}
// =====================================
// DRAW SHIELD
// =====================================

function drawShield(ctx, x, y, size) {

    ctx.save();

    ctx.translate(x, y);

    // Shield
    ctx.beginPath();

    ctx.moveTo(size * 0.5, 0);

    ctx.lineTo(size, size * 0.20);

    ctx.lineTo(size * 0.85, size * 0.85);

    ctx.quadraticCurveTo(
        size * 0.50,
        size * 1.25,
        size * 0.15,
        size * 0.85
    );

    ctx.lineTo(0, size * 0.20);

    ctx.closePath();

    ctx.fillStyle = "#FFFFFF";
    ctx.fill();

    // Check
    ctx.beginPath();

    ctx.strokeStyle = "#D90429";
    ctx.lineWidth = size * 0.12;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    ctx.moveTo(size * 0.28, size * 0.60);
    ctx.lineTo(size * 0.45, size * 0.78);
    ctx.lineTo(size * 0.74, size * 0.38);

    ctx.stroke();

    ctx.restore();

}
// =====================================================
// BUILD WATERMARK V2
// =====================================================

async function buildWatermarkV2() {

    if (!app.photoData) {

        throw new Error("No photo captured.");

    }

    status.innerHTML = "🖼️ Preparing Watermark...";

    return new Promise((resolve, reject) => {

        const img = new Image();

        img.onload = function () {

            canvas.width = img.width;
            canvas.height = img.height;

            const ctx = canvas.getContext("2d");

            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = "high";

            // =====================================
            // DRAW PHOTO
            // =====================================

            ctx.drawImage(
                img,
                0,
                0,
                canvas.width,
                canvas.height
            );

            // =====================================
            // DATE / TIME
            // =====================================

            const now = new Date();

            const dateText = now.toLocaleDateString("en-PH", {

                year: "numeric",
                month: "long",
                day: "numeric"

            });

            const timeText = now.toLocaleTimeString("en-PH", {

                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit"

            });

 // =====================================
// RESPONSIVE SIZE
// =====================================

const scale = canvas.width / 640;

const cardWidth = Math.min(400, canvas.width * 0.64);

const titleFont = Math.round(18 * scale);

const subtitleFont = Math.round(10 * scale);

const bodyFont = Math.round(14 * scale);

const smallFont = Math.round(12 * scale);

const lineHeight = Math.round(19 * scale);

            // =====================================
            // CARD SIZE
            // =====================================

            const cardHeight = 205;

            const cardX = 3;

            const cardY =
                canvas.height - cardHeight - 3;

            // =====================================
            // ROUNDED GLASS CARD
            // =====================================

            ctx.save();

            ctx.beginPath();

            const radius = 18;

            ctx.moveTo(cardX + radius, cardY);

            ctx.lineTo(cardX + cardWidth - radius, cardY);

            ctx.quadraticCurveTo(
                cardX + cardWidth,
                cardY,
                cardX + cardWidth,
                cardY + radius
            );

            ctx.lineTo(
                cardX + cardWidth,
                cardY + cardHeight - radius
            );

            ctx.quadraticCurveTo(
                cardX + cardWidth,
                cardY + cardHeight,
                cardX + cardWidth - radius,
                cardY + cardHeight
            );

            ctx.lineTo(
                cardX + radius,
                cardY + cardHeight
            );

            ctx.quadraticCurveTo(
                cardX,
                cardY + cardHeight,
                cardX,
                cardY + cardHeight - radius
            );

            ctx.lineTo(
                cardX,
                cardY + radius
            );

            ctx.quadraticCurveTo(
                cardX,
                cardY,
                cardX + radius,
                cardY
            );

            ctx.closePath();

            // 45% opacity
            ctx.fillStyle = "rgba(20,20,20,0.25)";
            ctx.shadowColor = "rgba(0,0,0,.35)";
            ctx.shadowBlur = 18;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 6;

            ctx.fill();

            ctx.shadowBlur = 0;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 0;
            ctx.shadowColor = "transparent";

            // White Border
            ctx.strokeStyle = "rgba(255,255,255,.30)";
            ctx.lineWidth = 1.5;
            ctx.stroke();

            ctx.restore();

             // =====================================
            // START POSITION
            // =====================================

            let y = cardY + 26;

            const left = cardX + 18;

            const right =
                cardX + cardWidth * 0.55;

            // =====================================
            // HEADER
            // =====================================

            ctx.textBaseline = "top";

            ctx.fillStyle = "#FFFFFF";

            ctx.font = `bold ${titleFont}px Arial`;

            ctx.fillText(
                "RJAB CORPORATION",
                left,
                y
            );

            y += 20;

            ctx.font = `${subtitleFont}px Arial`;

            ctx.fillStyle = "#E5E7EB";

            ctx.fillText(
                "ATTENDANCE VERIFICATION",
                left,
                y
            );

            y += 12;

            // =====================================
            // DIVIDER
            // =====================================

            ctx.strokeStyle = "rgba(255,255,255,.10)";
            ctx.lineWidth = 1;

            ctx.beginPath();

            ctx.moveTo(
                left,
                y
            );

            ctx.lineTo(
                cardX + cardWidth - 18,
                y
            );

            ctx.stroke();

            y += 16;

            // =====================================
            // DATE | TIME
            // =====================================

            ctx.font = `${bodyFont}px Arial`;

            y = drawInfoRow(

                ctx,

                "📅",
                dateText,

                "🕒",
                timeText,

                y,

                left,

                right,

                bodyFont

            );

            // =====================================
// Employee identity is resolved by the authenticated portal from the ticket.

            // TYPE | GPS
            // =====================================

            y = drawInfoRow(

                ctx,

                "📋",
                app.type || "-",

                "🎯",
                "±" + (app.accuracy || "-") + " m",

                y,

                left,

                right,

                bodyFont

            );

            y += 6;

            // =====================================
            // LOCATION TITLE
            // =====================================

            ctx.font = `bold ${bodyFont}px Arial`;
            ctx.fillStyle = "#FFFFFF";

            ctx.fillText(
                "📍 Location",
                left,
                y
            );

            y += lineHeight;

            // =====================================
            // ADDRESS
            // =====================================

            ctx.font = `${smallFont}px Arial`;
            ctx.fillStyle = "#F3F4F6";

            const address = (app.address || "Unknown Address")
                .split(",")
                .slice(0, 5)
                .join(", ");

            y = drawWrappedText(
                ctx,
                address,
                left,
                y,
                cardWidth - 36,
                15
            );

            y += 10;

            // =====================================
            // FOOTER
            // =====================================

            const footerHeight = 22;

            ctx.fillStyle = "#D90429";

            ctx.beginPath();

            ctx.moveTo(cardX, cardY + cardHeight - footerHeight);

            ctx.lineTo(cardX + cardWidth, cardY + cardHeight - footerHeight);

            ctx.lineTo(cardX + cardWidth, cardY + cardHeight - 18);

            ctx.quadraticCurveTo(
                cardX + cardWidth,
                cardY + cardHeight,
                cardX + cardWidth - 18,
                cardY + cardHeight
            );

            ctx.lineTo(
                cardX + 18,
                cardY + cardHeight
            );

            ctx.quadraticCurveTo(
                cardX,
                cardY + cardHeight,
                cardX,
                cardY + cardHeight - 18
            );

            ctx.closePath();

            ctx.fill();

            ctx.fillStyle = "#FFFFFF";
            ctx.textBaseline = "middle";

            // =====================================
            // FOOTER CONTENT
            // =====================================

            const footerText = "CAPTURED USING RJAB ATTENDANCE CAMERA";

            ctx.font = "bold 9px Arial";

            // Sukatin ang text
            const textWidth = ctx.measureText(footerText).width;

            // Shield size
            const shieldSize = 14;

            // Space sa pagitan
            const gap = 10;

            // Total width ng shield + gap + text
            const totalWidth =
                shieldSize +
                gap +
                textWidth;

            // Simula ng group
            const startX =
                cardX + (cardWidth - totalWidth) / 2;

            // Vertical center
            const centerY =
                cardY + cardHeight - (footerHeight / 2);

            drawShield(
                ctx,
                startX,
                centerY - (shieldSize / 2) + 1,
                shieldSize
            );
            ctx.fillStyle = "#FFFFFF";

            ctx.font = "bold 9px Arial";

            ctx.textAlign = "left";
            ctx.textBaseline = "middle";

            ctx.fillText(
                footerText,
                startX + shieldSize + gap,
                centerY
            );

            // Restore defaults
            ctx.textAlign = "left";
            ctx.textBaseline = "top";

            // =====================================
            // EXPORT
            // =====================================

            app.photoData = canvas.toDataURL(
                "image/jpeg",
                0.95
            );

            preview.src = app.photoData;

            resolve();

        };

        img.onerror = function () {

            reject("Unable to load image.");

        };

        img.src = app.photoData;

    });

}
// ============================
// BUILD WATERMARK
// ============================

async function buildWatermark() {

    if (!app.photoData) {
        throw new Error("No photo captured.");
    }

    status.innerHTML = "🖼️ Preparing Watermark...";

    return new Promise((resolve, reject) => {

        const img = new Image();

        img.onload = function () {

            // =====================================
            // LAYOUT CONFIGURATION
            // =====================================

            const layout = {

                labelX: 25,
                valueX: 180,

                titleTop: 38,
                subtitleTop: 88,

                dividerTop: 125,

                startY: 145,

                rowHeight: 36,

                locationGap: 44,

                addressLineHeight: 28,

                verificationGap: 30

            };

            // =====================================
            // MEASURE ADDRESS
            // =====================================

            const measureCanvas = document.createElement("canvas");
            const measureCtx = measureCanvas.getContext("2d");

            measureCtx.font = "bold 19px Arial";

            const addressLines = getWrappedLineCount(
                measureCtx,
                app.address || "",
                img.width - layout.valueX - 25
            );

            const fixedContent =
    layout.startY +
    (layout.rowHeight * 6) +
    layout.locationGap;

         const footerHeight =
    fixedContent +
    (addressLines * layout.addressLineHeight) +
    layout.verificationGap +
    70;

            canvas.width = img.width;
            canvas.height = img.height + footerHeight;

            const ctx = canvas.getContext("2d");

            ctx.textBaseline = "top";

            ctx.font = "20px Arial";

                        const now = new Date();

            const dateText = now.toLocaleDateString("en-PH", {
                year: "numeric",
                month: "long",
                day: "numeric"
            });

            const timeText = now.toLocaleTimeString("en-PH", {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit"
            });

            // =====================================
            // DRAW PHOTO
            // =====================================

            ctx.drawImage(img, 0, 0);

            // =====================================
            // PREMIUM BACKGROUND
            // =====================================

            const gradient = ctx.createLinearGradient(
                0,
                img.height,
                0,
                canvas.height
            );

            gradient.addColorStop(0.00, "#120000");
            gradient.addColorStop(0.25, "#2A0000");
            gradient.addColorStop(0.55, "#4D0000");
            gradient.addColorStop(1.00, "#7D0000");

            ctx.fillStyle = gradient;

            ctx.fillRect(
                0,
                img.height,
                canvas.width,
                footerHeight
            );

            // =====================================
            // TOP ACCENT
            // =====================================

            ctx.fillStyle = "#FF3B3B";

            ctx.fillRect(
                0,
                img.height,
                canvas.width,
                5
            );

            // =====================================
            // HEADER
            // =====================================

            ctx.textAlign = "center";

            ctx.fillStyle = "#FFFFFF";
            ctx.font = "bold 40px Arial";

            ctx.fillText(
                "RJAB CORPORATION",
                canvas.width / 2,
                img.height + layout.titleTop
            );

            ctx.fillStyle = "#F8D7DA";
            ctx.font = "bold 24px Arial";

            ctx.fillText(
                "ATTENDANCE VERIFICATION",
                canvas.width / 2,
                img.height + layout.subtitleTop
            );

            // =====================================
            // DIVIDER
            // =====================================

            ctx.strokeStyle = "#FF4A4A";
            ctx.lineWidth = 2;

            ctx.beginPath();

            ctx.moveTo(
                30,
                img.height + layout.dividerTop
            );

            ctx.lineTo(
                canvas.width - 30,
                img.height + layout.dividerTop
            );

            ctx.stroke();

            ctx.textAlign = "left";

            let y = img.height + layout.startY;

                        // =====================================
            // INFORMATION TABLE
            // =====================================

            function drawRow(label, value) {

                ctx.fillStyle = "#FFDCDC";
                ctx.font = "19px Arial";

                ctx.fillText(
                    label,
                    layout.labelX,
                    y
                );

                ctx.fillStyle = "#FFFFFF";
                ctx.font = "bold 19px Arial";

                ctx.fillText(
    ": " + (value || "-"),
    layout.valueX,
    y
);

                y += layout.rowHeight;

            }

            drawRow("Date", dateText);
            drawRow("Time", timeText);
            drawRow("Type", app.type);
            drawRow("Accuracy", "±" + app.accuracy + " m");

            y += 8;

            // =====================================
            // LOCATION
            // =====================================

            ctx.fillStyle = "#FFDCDC";
            ctx.font = "19px Arial";

            ctx.fillText(
    "Location",
    layout.labelX,
    y + 2
);

            ctx.fillStyle = "#FFFFFF";
            ctx.font = "bold 19px Arial";

            y = drawWrappedText(

                ctx,

                ": " + (app.address || "Unknown Address"),

                layout.valueX,

                y,

                canvas.width - layout.valueX - 25,

                layout.addressLineHeight

            );
            // Bottom padding bago ang footer
            y += 12;

             // =====================================
            // VERIFICATION FOOTER
            // =====================================

            y += layout.verificationGap;

            ctx.strokeStyle = "rgba(255,255,255,0.25)";
            ctx.lineWidth = 1;

            ctx.beginPath();

            ctx.moveTo(
                25,
                y
            );

            ctx.lineTo(
                canvas.width - 25,
                y
            );

            ctx.stroke();

            y += 25;

            ctx.textAlign = "center";

// Verification Text
ctx.fillStyle = "#FFDCDC";
ctx.font = "bold 16px Arial";

ctx.fillText(
    "VERIFIED USING RJAB CORPORATION CAMERA SYSTEM",
    canvas.width / 2,
    y
);

// =====================================
// COPYRIGHT
// =====================================

const currentYear = new Date().getFullYear();

ctx.fillStyle = "rgba(255,255,255,0.55)";
ctx.font = "13px Arial";

ctx.fillText(
    "© " + currentYear + " RJAB CORPORATION",
    canvas.width / 2,
    y + 22
);

            // =====================================
            // EXPORT IMAGE
            // =====================================

            app.photoData = canvas.toDataURL(
                "image/jpeg",
                0.95
            );

            preview.src = app.photoData;

            resolve();

        };

        img.onerror = function () {

            reject("Unable to load image.");

        };

        img.src = app.photoData;

    });

}

// =====================================================
// RETURN CAPTURE TO AUTHENTICATED ATTENDANCE PORTAL
// =====================================================

async function uploadPhoto() {
    if (uploadInProgress) {
        console.warn("Attendance capture handoff already in progress.");
        return;
    }

    if (!app.ticket || !attendanceParentWindow || !app.parentOrigin) {
        throw new Error("Secure attendance handoff is not initialized. Reopen the camera from the Employee Portal.");
    }

    if (!app.photoData || !app.latitude || !app.longitude) {
        throw new Error("The live photo or GPS coordinates are missing.");
    }

    uploadInProgress = true;
    setVerifyButton(false);
    status.innerHTML = "🔐 Sending secure capture to the RJAB Attendance Portal...";

    const payload = {
        action: app.type === "TIME IN" ? "TIME_IN" : "TIME_OUT",
        photoDataUrl: app.photoData,
        photoMimeType: "image/jpeg",
        latitude: app.latitude,
        longitude: app.longitude,
        gpsAccuracyMeters: app.accuracy,
        gpsTimestamp: app.gpsTimestamp || "",
        address: app.address || "Address not available",
        clientCapturedAt: app.clientCapturedAt || new Date().toISOString(),
        cameraQuality: app.scannerSkipped ? "SKIPPED" : "PASSED",
        cameraVersion: app.cameraVersion
    };

    const envelope = {
        type: "RJAB_ATTENDANCE_CAPTURE",
        version: 1,
        ticket: app.ticket,
        payload: payload
    };

    const ack = await new Promise((resolve, reject) => {
        const timeoutId = setTimeout(() => {
            if (pendingAttendanceAck) pendingAttendanceAck = null;
            reject(new Error("No confirmation was received from the Attendance Portal. Do not submit another attendance action yet. Return to the portal and check whether the request was saved."));
        }, 45000);

        pendingAttendanceAck = { resolve, reject, timeoutId };

        try {
            attendanceParentWindow.postMessage(envelope, app.parentOrigin);
        } catch (error) {
            clearTimeout(timeoutId);
            pendingAttendanceAck = null;
            reject(new Error("Unable to send the photo to the Attendance Portal: " + error.message));
        }
    });

    if (!ack || ack.ok !== true) {
        throw new Error("Attendance Portal did not confirm the capture.");
    }

    status.innerHTML = `
        <div style="color:#22c55e;font-weight:bold;font-size:20px">
            ✅ Capture Received by Attendance Portal
        </div>
        <div style="margin-top:10px">
            Your ${app.type.toLowerCase()} photo and GPS details have been handed back for server validation.
        </div>
        <div style="margin-top:8px;font-size:12px;opacity:.8">
            You may return to the RJAB Attendance Portal.
        </div>
    `;

    preview.style.opacity = "0.9";
    preview.style.pointerEvents = "none";
    captureBtn.disabled = true;
    retakeBtn.disabled = true;
    useBtn.disabled = true;
    document.querySelector(".buttons").style.display = "none";
    video.hidden = true;
    guide.hidden = true;

    // Clear the image from this page only after the parent acknowledged receipt.
    app.photoData = "";
    preview.src = "";

    // A popup opened by the portal may close itself; if blocked, the success
    // message remains visible and the employee can close this tab manually.
    setTimeout(() => { try { window.close(); } catch (e) {} }, 900);
}

// ============================
// VERIFY BUTTON STATE
// ============================

function setVerifyButton(enabled) {

    useBtn.disabled = !enabled;

    useBtn.style.opacity = enabled ? "1" : "0.35";

    useBtn.style.pointerEvents = enabled ? "auto" : "none";

}
// ============================
// USE PHOTO
// ============================

useBtn.onclick = async function () {

    if (uploadInProgress) {
        console.log("📸 Verification already in progress. Duplicate click blocked.");
        return;
    }

    setVerifyButton(false);

    controls.style.display = "none";

    // Hide Photo Quality Panel
    qualityStatus.style.display = "none";

    // Show Processing Panel
    status.style.display = "block";

    status.innerHTML = "📍 Getting GPS...";

    try {

        await getGPS();

        status.innerHTML = "🌍 Getting Address...";

        await getAddress();

        await buildWatermarkV2();

        await uploadPhoto();

    }

    catch (error) {

    console.error(error);

    uploadInProgress = false;

    alert("Operation Failed\n\n" + error);

    status.innerHTML = "❌ Operation Failed";

    // Unlock everything
    captureLocked = false;

    captureBtn.disabled = false;
    retakeBtn.disabled = false;

    setVerifyButton(true);

    controls.style.display = "flex";

    qualityStatus.style.display = "block";

    // Restore the correct status after 2 seconds
    setTimeout(function () {

        if (app.captured) {

            status.innerHTML = "📸 Ready to Upload";

        } else {

            status.innerHTML = "✅ Camera Ready";

        }

    }, 2000);

}

};

// ============================
// ATTENDANCE CAMERA STARTUP
// ============================

try {
    validateAttendanceLaunch();
    initFaceDetector();
    startCamera();
} catch (launchError) {
    console.error("Attendance camera launch rejected:", launchError);
    status.style.display = "block";
    status.innerHTML = "❌ " + launchError.message;
    captureBtn.hidden = true;
    captureBtn.disabled = true;
    retakeBtn.hidden = true;
    useBtn.hidden = true;
    video.hidden = true;
    const context = document.getElementById("attendanceContext");
    if (context) context.textContent = "PORTAL LAUNCH REQUIRED";
}
