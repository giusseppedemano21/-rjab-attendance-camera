/*
 * Example only: parent-side listener for the RJAB Attendance Camera fork.
 * Implement server-side ticket validation and authenticated saving in Apps Script.
 * Never trust the client action, client timestamp, or GPS accuracy without policy checks.
 */
const RJAB_CAMERA_ORIGIN = "https://YOUR_GITHUB_USERNAME.github.io";
let rjabCameraWindow = null;
let activeAttendanceTicket = null;

function openRjabAttendanceCamera(cameraBaseUrl, ticket, action) {
  const mode = action === "TIME_IN" ? "TIMEIN" : "TIMEOUT";
  const parentOrigin = window.location.origin;
  const url = new URL(cameraBaseUrl);
  url.searchParams.set("ticket", ticket);
  url.searchParams.set("mode", mode); // UI hint only
  url.searchParams.set("parentOrigin", parentOrigin);
  activeAttendanceTicket = ticket;
  rjabCameraWindow = window.open(url.toString(), "rjabAttendanceCamera", "popup,width=520,height=850");
  if (!rjabCameraWindow) throw new Error("Please allow popups for the RJAB Attendance Portal.");
}

window.addEventListener("message", async (event) => {
  if (event.origin !== RJAB_CAMERA_ORIGIN) return; // replace with exact origin
  if (!rjabCameraWindow || event.source !== rjabCameraWindow) return;
  const message = event.data || {};
  if (message.type !== "RJAB_ATTENDANCE_CAPTURE" || message.version !== 1) return;
  if (!activeAttendanceTicket || message.ticket !== activeAttendanceTicket) return;

  try {
    // Replace with authenticated google.script.run call. The backend must validate
    // and consume the one-time ticket before storing the capture. Do not log photoDataUrl.
    const saveResult = await saveAttendanceCaptureThroughAppsScript(message);
    rjabCameraWindow.postMessage({
      type: "RJAB_ATTENDANCE_CAPTURE_ACK",
      ticket: message.ticket,
      ok: true,
      message: "Attendance capture received."
    }, event.origin);
  } catch (error) {
    rjabCameraWindow.postMessage({
      type: "RJAB_ATTENDANCE_CAPTURE_ACK",
      ticket: message.ticket,
      ok: false,
      message: "Unable to save the attendance capture. Please try again or contact the Admin."
    }, event.origin);
  }
});
