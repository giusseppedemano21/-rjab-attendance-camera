# RJAB Attendance Camera — Phase 2G.1 AI Diagnostic Build

## Scope of this build

This package adds an **optional, browser-only test panel** to the existing RJAB Attendance Camera. The original ticket validation, TIME IN/TIME OUT capture, photo-quality checks, watermark, GPS/address collection, portal `postMessage` handoff, and server acknowledgement are retained.

The optional panel combines:
- MediaPipe Face Landmarker for 3D/2D facial landmarks;
- EdgeFace-XXS for 512-value face embeddings;
- MiniFASNet V2 and V1SE for experimental PAD scores.

The panel supports an in-memory temporary reference, same/different-person similarity tests, and five-frame PAD diagnostics. It does **not** compare against the employee's saved Google Drive template or make attendance decisions. No image, embedding, PAD score, or result is added to the TIME IN/TIME OUT payload. The temporary reference is held only in JavaScript memory and is cleared when the page reloads/closes or the Clear button is pressed.

## Files to deploy to the existing GitHub repository root

Replace the corresponding camera source files with the files from this package, then upload these three model assets to the same repository root:
- `edgeface_xxs.onnx`
- `MiniFASNetV2.onnx`
- `MiniFASNetV1SE.onnx`

Keep existing camera image assets (`cameraback.jpg`, `cameraback1.jpg`) and other existing root files. Do not upload the folder as a nested directory; `index.html` and `script.js` must remain at the repository root.

## Apps Script pilot switch

1. Replace the current `AttendanceCameraIntegration.gs` with `AttendanceCameraIntegration_PHASE2G1_AI_TEST.gs` (or apply its small URL-construction change to the existing file).
2. In Apps Script → Project Settings → Script Properties, add:
   - Property: `ATTENDANCE_AI_TEST_MODE`
   - Value: `1`
3. Save and redeploy the Apps Script Web App if your project requires a new deployment version.
4. Launch TIME IN or TIME OUT from the authenticated Employee Portal. The AI diagnostics panel should appear below the camera preview.
5. When testing is finished, set `ATTENDANCE_AI_TEST_MODE` to `0` or delete the property. Normal launches then hide the panel and do not download ONNX Runtime or the three AI models.

## Suggested controlled test

Use only consenting test participants. Keep the existing attendance capture process separate from this diagnostic:

1. Start the camera and wait until all model checksums have passed and the AI panel says Ready.
2. Capture a temporary reference using one consenting tester (three samples).
3. Compare that same tester several times and record the cosine scores.
4. Without clearing the reference, ask a different consenting tester to compare and record the scores.
5. Repeat under normal indoor/outdoor light and on iPhone Safari / Android Chrome / desktop Chrome.
6. Run PAD diagnostics separately with a live consenting face, a printed photo, and a replay on another screen. Record V2 and V1SE medians. Do not treat the model result as a guarantee against spoofing.
7. Clear the reference when finished.

The prototype's 0.80 PAD score flag and any cosine similarity values are exploratory only. This build deliberately applies no identity accept/reject threshold and does not authorize an attendance punch.

## Important limitations

- The temporary reference is generated in the browser, not read from the encrypted employee template in the Worker.
- Browser-generated embeddings and PAD results can be forged on a compromised or modified client.
- The face alignment points are an approximation and require calibration using RJAB camera samples.
- The MiniFASNet crop logic is adapted from the earlier prototype; its scores must be validated on actual RJAB devices and camera frames.
- This test build does not modify the Cloudflare Worker and does not activate any pending biometric template.
- Attendance captures remain in the existing pending identity-verification state. Do not use these diagnostics for payroll or as official attendance verification.

## Model checksums included in the code

- EdgeFace-XXS: `dc674de4cbc77fa0bf9a82d5149558ab8581d82a2cd3bb60f28fd1a5d3ff8a2f`
- MiniFASNet V2: `b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907`
- MiniFASNet V1SE: `ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676`

See `THIRD_PARTY_NOTICES.md` and preserve upstream license notices when redistributing the model assets.
