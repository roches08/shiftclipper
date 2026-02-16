// Minimal vanilla JS client for ShiftClipper

const API_BASE = "";

function $(id) {
  return document.getElementById(id);
}

function must(id) {
  const el = $(id);
  if (!el) {
    throw new Error(
      `UI element #${id} not found. Make sure you copied web/index.html and web/app.js to /workspace/shiftclipper/Projects/web/`
    );
  }
  return el;
}

function fmtTime(s) {
  if (s == null || isNaN(s)) return "—";
  const m = Math.floor(s / 60);
  const ss = Math.floor(s % 60);
  return `${m}:${String(ss).padStart(2, "0")}`;
}

async function api(path, opts = {}) {
  const res = await fetch(`${API_BASE}${path}`, opts);
  if (!res.ok) {
    let text = "";
    try {
      text = await res.text();
    } catch (e) {}
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res.text();
}

let state = {
  job_id: null,
  status: null,
  setup: null,
  clicks: [],
  video_url: null,
  proxy_url: null,
  duration: null,
};

function setStatus(msg) {
  must("status").textContent = msg;
}

function setJobId(id) {
  state.job_id = id;
  must("jobId").textContent = id || "—";
}

function clearClicks() {
  state.clicks = [];
  renderClicks();
}

function renderClicks() {
  const ul = must("clicksList");
  ul.innerHTML = "";
  state.clicks.forEach((c, idx) => {
    const li = document.createElement("li");
    li.textContent = `${idx + 1}. t=${c.t.toFixed(2)} x=${c.x.toFixed(
      3
    )} y=${c.y.toFixed(3)}`;
    ul.appendChild(li);
  });
  must("clickCount").textContent = String(state.clicks.length);
}

function getVideoTime() {
  const v = must("vid");
  return v.currentTime || 0;
}

function getVideoXY(evt) {
  const v = must("vid");
  const rect = v.getBoundingClientRect();
  const x = (evt.clientX - rect.left) / rect.width;
  const y = (evt.clientY - rect.top) / rect.height;
  return { x: Math.min(1, Math.max(0, x)), y: Math.min(1, Math.max(0, y)) };
}

function enableControls(enabled) {
  must("btnUpload").disabled = !enabled;
  must("btnRun").disabled = !enabled;
  must("btnReset").disabled = !enabled;
  must("btnSaveSetup").disabled = !enabled;
}

async function createJob() {
  setStatus("Creating job...");
  const resp = await api("/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ camera_mode: must("cameraMode").value || "broadcast" }),
  });
  setJobId(resp.job_id);
  setStatus("Job created. Upload a video.");
  enableControls(true);
}

async function uploadVideo() {
  const job_id = state.job_id;
  if (!job_id) throw new Error("Create a job first.");

  const f = must("file").files[0];
  if (!f) throw new Error("Choose a video file first.");

  setStatus("Uploading video...");
  const fd = new FormData();
  fd.append("file", f);

  await api(`/jobs/${job_id}/upload`, { method: "POST", body: fd });
  setStatus("Uploaded. Waiting for proxy...");
  await pollStatus(true);
}

async function saveSetup() {
  const job_id = state.job_id;
  if (!job_id) throw new Error("Create a job first.");

  const payload = {
    player_number: must("playerNumber").value || "",
    jersey_color: must("jerseyColor").value || "#203524",
    opponent_color: must("opponentColor").value || "#ffffff",
    extend_sec: parseFloat(must("extendSec").value || "2"),
    verify_mode: must("verifyMode").checked,
    clicks: state.clicks,
  };

  await api(`/jobs/${job_id}/setup`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  setStatus("Setup saved. Click RUN.");
}

async function runJob() {
  const job_id = state.job_id;
  if (!job_id) throw new Error("Create a job first.");

  setStatus("Queueing job...");
  await api(`/jobs/${job_id}/run`, { method: "POST" });
  await pollStatus(true);
}

async function resetAll() {
  setJobId(null);
  state.status = null;
  state.setup = null;
  state.video_url = null;
  state.proxy_url = null;
  state.duration = null;

  clearClicks();
  must("vid").removeAttribute("src");
  must("vid").load();
  must("clips").innerHTML = "";
  must("combined").innerHTML = "";
  enableControls(false);
  setStatus("Reset. Create a new job.");
}

function renderResults(status) {
  const clipsDiv = must("clips");
  clipsDiv.innerHTML = "";
  const combinedDiv = must("combined");
  combinedDiv.innerHTML = "";

  if (status && Array.isArray(status.clips) && status.clips.length) {
    status.clips.forEach((c, i) => {
      const a = document.createElement("a");
      a.href = c.url;
      a.target = "_blank";
      a.textContent = `▶ Clip ${String(i + 1).padStart(2, "0")} (${fmtTime(
        c.start
      )}–${fmtTime(c.end)})`;
      const p = document.createElement("div");
      p.appendChild(a);
      clipsDiv.appendChild(p);
    });
  }

  if (status && status.combined_url) {
    const a = document.createElement("a");
    a.href = status.combined_url;
    a.target = "_blank";
    a.textContent = "▶ Combined video";
    combinedDiv.appendChild(a);
  }
}

async function pollStatus(force = false) {
  const job_id = state.job_id;
  if (!job_id) return;

  const st = await api(`/jobs/${job_id}/status`);
  state.status = st;

  must("statusJson").textContent = JSON.stringify(st, null, 2);

  // If proxy exists, load it in the player
  if (st.proxy_url && st.proxy_url !== state.proxy_url) {
    state.proxy_url = st.proxy_url;
    const v = must("vid");
    v.src = st.proxy_url;
    v.load();
  }

  if (st.status) setStatus(`${st.status} (${st.progress ?? 0}%) — ${st.message ?? ""}`);

  if (st.status === "done" || st.status === "error") {
    renderResults(st);
    return;
  }

  // keep polling while running/queued
  if (force || ["queued", "running", "uploaded", "ready"].includes(st.status)) {
    setTimeout(() => pollStatus(true), 1500);
  }
}

function wire() {
  enableControls(false);
  setStatus("Ready. Click Create Job.");

  must("btnCreate").addEventListener("click", async () => {
    try {
      await createJob();
    } catch (e) {
      console.error(e);
      alert(String(e));
      setStatus(String(e));
    }
  });

  must("btnUpload").addEventListener("click", async () => {
    try {
      await uploadVideo();
    } catch (e) {
      console.error(e);
      alert(String(e));
      setStatus(String(e));
    }
  });

  must("btnSaveSetup").addEventListener("click", async () => {
    try {
      await saveSetup();
    } catch (e) {
      console.error(e);
      alert(String(e));
      setStatus(String(e));
    }
  });

  must("btnRun").addEventListener("click", async () => {
    try {
      await runJob();
    } catch (e) {
      console.error(e);
      alert(String(e));
      setStatus(String(e));
    }
  });

  must("btnReset").addEventListener("click", async () => {
    try {
      await resetAll();
    } catch (e) {
      console.error(e);
      alert(String(e));
    }
  });

  // Click-to-seed
  must("vid").addEventListener("click", (evt) => {
    const t = getVideoTime();
    const { x, y } = getVideoXY(evt);
    state.clicks.push({ t, x, y });
    renderClicks();
  });

  // keep status updated
  setInterval(() => pollStatus(false), 2500);
}

window.addEventListener("DOMContentLoaded", () => {
  try {
    wire();
  } catch (e) {
    console.error(e);
    alert(String(e));
  }
});

