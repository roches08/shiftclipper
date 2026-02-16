cat > web/app.js <<'JS'
let JOB_ID = null;
let CLICKS = [];

const el = (id) => document.getElementById(id);

const btnCreate = el("btnCreate");
const btnUpload = el("btnUpload");
const btnSaveSetup = el("btnSaveSetup");
const btnRun = el("btnRun");
const btnReset = el("btnReset");

const jobIdEl = el("jobId");
const jobStatusEl = el("jobStatus");
const jobMsgEl = el("jobMsg");

const progressBar = el("progressBar");
const progressText = el("progressText");

const fileInput = el("fileInput");
const video = el("video");

const cameraMode = el("cameraMode");
const playerNumber = el("playerNumber");
const jerseyColor = el("jerseyColor");
const oppColor = el("oppColor");
const extendSec = el("extendSec");
const verifyMode = el("verifyMode");

const clickCount = el("clickCount");
const clickDump = el("clickDump");

const clipsEl = el("clips");
const statusJsonEl = el("statusJson");

function setBadge(text) {
  jobStatusEl.textContent = text || "idle";
}

function setProgress(pct, msg="") {
  const v = Math.max(0, Math.min(100, Number(pct || 0)));
  progressBar.value = v;
  progressText.textContent = `Progress: ${v}% ${msg ? "— " + msg : ""}`;
}

function enableControls() {
  btnUpload.disabled = !JOB_ID;
  btnReset.disabled = !JOB_ID;
  btnSaveSetup.disabled = !JOB_ID;
  // Run only when uploaded+proxy_ready OR at least uploaded (worker will make proxy)
  // We'll refine this using status polling.
}

function resetStateUI() {
  JOB_ID = null;
  CLICKS = [];
  jobIdEl.textContent = "—";
  setBadge("idle");
  jobMsgEl.textContent = "";
  setProgress(0, "");
  clickCount.textContent = "0";
  clickDump.textContent = "";
  clipsEl.innerHTML = "";
  statusJsonEl.textContent = "{}";
  btnUpload.disabled = true;
  btnSaveSetup.disabled = true;
  btnRun.disabled = true;
  btnReset.disabled = true;
  video.removeAttribute("src");
  video.load();
}

function renderClicks() {
  clickCount.textContent = String(CLICKS.length);
  if (!CLICKS.length) {
    clickDump.textContent = "";
    return;
  }
  clickDump.textContent = CLICKS.map((c, i) => {
    return `#${i+1}: t=${c.t.toFixed(2)}s x=${c.x.toFixed(3)} y=${c.y.toFixed(3)}`;
  }).join(" | ");
}

async function apiJSON(path, opts={}) {
  const res = await fetch(path, opts);
  const txt = await res.text();
  let data = null;
  try { data = txt ? JSON.parse(txt) : null; } catch { data = txt; }
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} ${JSON.stringify(data)}`);
  }
  return data;
}

function xhrUpload(url, file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const fd = new FormData();
    fd.append("file", file);

    xhr.open("POST", url, true);

    xhr.upload.onprogress = (evt) => {
      if (evt.lengthComputable && onProgress) {
        const pct = Math.round((evt.loaded / evt.total) * 100);
        onProgress(pct);
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.responseText ? JSON.parse(xhr.responseText) : {});
      } else {
        reject(new Error(`HTTP ${xhr.status}: ${xhr.responseText}`));
      }
    };

    xhr.onerror = () => reject(new Error("Upload failed (network error)."));
    xhr.send(fd);
  });
}

async function pollStatusOnce() {
  if (!JOB_ID) return;

  const st = await apiJSON(`/jobs/${JOB_ID}/status`);
  statusJsonEl.textContent = JSON.stringify(st, null, 2);

  setBadge(st.status || st.stage || "unknown");
  setProgress(st.progress ?? 0, st.message || "");

  jobMsgEl.textContent = st.message || "";

  // Update video src to proxy if available
  if (st.proxy_url && video.src.indexOf(st.proxy_url) === -1) {
    video.src = st.proxy_url + `?ts=${Date.now()}`;
    video.load();
  }

  // Buttons
  const uploaded = (st.status === "uploaded" || st.stage === "uploaded" || st.video_path);
  btnSaveSetup.disabled = !uploaded;

  const canRun = uploaded; // allow run after upload; worker handles proxy generation too
  btnRun.disabled = !canRun;

  // Render clips
  if (st.clips && Array.isArray(st.clips)) {
    clipsEl.innerHTML = "";
    st.clips.forEach((c, idx) => {
      const a = document.createElement("a");
      a.href = c.url;
      a.textContent = `▶ Clip ${String(idx+1).padStart(2,"0")} (${c.start.toFixed(2)}–${c.end.toFixed(2)}s)`;
      a.target = "_blank";
      clipsEl.appendChild(a);
    });
    if (st.combined_url) {
      const a = document.createElement("a");
      a.href = st.combined_url;
      a.textContent = "▶ Combined video";
      a.target = "_blank";
      clipsEl.appendChild(document.createElement("div"));
      clipsEl.appendChild(a);
    }
  }

  return st;
}

let pollTimer = null;
function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => {
    pollStatusOnce().catch(() => {});
  }, 1500);
}

// -------------------------
// Events
// -------------------------
btnCreate.addEventListener("click", async () => {
  try {
    setProgress(0, "Creating job…");
    const data = await apiJSON("/jobs", { method: "POST" });
    JOB_ID = data.job_id;
    jobIdEl.textContent = JOB_ID;
    btnReset.disabled = false;
    btnUpload.disabled = false;
    setBadge("created");
    setProgress(0, "Created.");
    startPolling();
  } catch (e) {
    console.error(e);
    alert(String(e));
  }
});

btnReset.addEventListener("click", async () => {
  // local reset only (keeps server files, but stops confusion)
  resetStateUI();
});

btnUpload.addEventListener("click", async () => {
  try {
    if (!JOB_ID) throw new Error("Create a job first.");
    const f = fileInput.files?.[0];
    if (!f) throw new Error("Choose a video file first.");

    setProgress(0, "Uploading…");
    btnUpload.disabled = true;

    await xhrUpload(`/jobs/${JOB_ID}/upload`, f, (pct) => {
      // Upload progress (0-100)
      setProgress(Math.min(9, Math.floor(pct * 0.09)), `Uploading… ${pct}%`);
    });

    // after upload, status polling will show uploaded/proxy
    setProgress(5, "Uploaded.");
    await pollStatusOnce();
    btnSaveSetup.disabled = false;
    btnRun.disabled = false;
  } catch (e) {
    console.error(e);
    alert(String(e));
    btnUpload.disabled = false;
  }
});

btnSaveSetup.addEventListener("click", async () => {
  try {
    if (!JOB_ID) throw new Error("Create a job first.");

    const payload = {
      camera_mode: cameraMode.value,
      player_number: playerNumber.value || "",
      jersey_color: jerseyColor.value,
      opponent_color: oppColor.value,
      extend_sec: Number(extendSec.value || 2),
      verify_mode: !!verifyMode.checked,
      clicks: CLICKS
    };

    await apiJSON(`/jobs/${JOB_ID}/setup`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    setProgress(progressBar.value, "Setup saved.");
    await pollStatusOnce();
  } catch (e) {
    console.error(e);
    alert(String(e));
  }
});

btnRun.addEventListener("click", async () => {
  try {
    if (!JOB_ID) throw new Error("Create a job first.");

    setProgress(10, "Queueing…");
    btnRun.disabled = true;

    await apiJSON(`/jobs/${JOB_ID}/run`, { method: "POST" });
    await pollStatusOnce();
  } catch (e) {
    console.error(e);
    alert(String(e));
    btnRun.disabled = false;
  }
});

// Click-to-seed (video)
video.addEventListener("click", (evt) => {
  if (!JOB_ID) return;

  const rect = video.getBoundingClientRect();
  const x = (evt.clientX - rect.left) / rect.width;
  const y = (evt.clientY - rect.top) / rect.height;
  const t = video.currentTime || 0;

  // cap at 8 just so you can't accidentally spam
  if (CLICKS.length >= 8) return;

  CLICKS.push({ t, x, y });
  renderClicks();
});

// init
resetStateUI();
JS

