// ShiftClipper MVP - Web UI
// - Relative URLs (RunPod proxy-safe)
// - Upload + click seeds
// - Save setup writes setup.json on server
// - Run triggers worker + polling results

let JOB_ID = null;
let VIDEO_EL = null;
let CLICK_SEEDS = [];

function $(id) { return document.getElementById(id); }

function api(path) {
  // Always relative (proxy-safe)
  return path.startsWith("/") ? path : `/${path}`;
}

function setStatus(msg) {
  const el = $("statusText");
  if (el) el.textContent = msg;
}

function setJobId(id) {
  JOB_ID = id;
  const el = $("jobIdText");
  if (el) el.textContent = id || "(none)";
}

function getFormValue(id, fallback = "") {
  const el = $(id);
  if (!el) return fallback;
  if (el.type === "checkbox") return !!el.checked;
  return el.value ?? fallback;
}

function buildSetupPayload() {
  const camera_mode = getFormValue("cameraMode", "broadcast");
  const player_number = getFormValue("playerNumber", "");
  const jersey_color = getFormValue("jerseyColor", "#112d27");
  const extend_sec = parseFloat(getFormValue("extendSec", "5")) || 5;
  const verify_mode = !!getFormValue("verifyMode", false);

  // Advanced knobs (optional; safe defaults match tasks.py)
  const min_clip_len = parseFloat(getFormValue("minClipLen", "20")) || 20;
  const pre_roll = parseFloat(getFormValue("preRoll", "4")) || 4;
  const post_roll = parseFloat(getFormValue("postRoll", "1.5")) || 1.5;
  const sticky_seconds = parseFloat(getFormValue("stickySeconds", "1.5")) || 1.5;
  const detect_stride = parseInt(getFormValue("detectStride", "3"), 10) || 3;

  return {
    camera_mode,
    player_number,
    jersey_color,
    extend_sec,
    verify_mode,
    min_clip_len,
    pre_roll,
    post_roll,
    sticky_seconds,
    detect_stride,
    clicks: CLICK_SEEDS.slice(),
    clicks_count: CLICK_SEEDS.length,
  };
}

function addClickSeed(t, x, y) {
  CLICK_SEEDS.push({ t, x, y });
  renderSeeds();
}

function clearSeeds() {
  CLICK_SEEDS = [];
  renderSeeds();
}

function renderSeeds() {
  const el = $("seedList");
  if (!el) return;
  el.innerHTML = "";
  CLICK_SEEDS.forEach((c, i) => {
    const li = document.createElement("li");
    li.textContent = `#${i + 1}: t=${c.t.toFixed(3)} x=${c.x.toFixed(3)} y=${c.y.toFixed(3)}`;
    el.appendChild(li);
  });
  const cnt = $("seedCount");
  if (cnt) cnt.textContent = String(CLICK_SEEDS.length);
}

async function createJob() {
  setStatus("Creating job…");
  const res = await fetch(api("/jobs"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: "job" })
  });
  if (!res.ok) {
    setStatus("Create job failed.");
    const txt = await res.text();
    throw new Error(txt);
  }
  const data = await res.json();
  setJobId(data.job_id);
  setStatus("Job created. Upload a video.");
}

async function uploadVideo(file) {
  if (!JOB_ID) throw new Error("No job. Click Create Job first.");

  setStatus("Uploading…");
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(api(`/jobs/${JOB_ID}/upload`), {
    method: "POST",
    body: form
  });

  if (!res.ok) {
    const txt = await res.text();
    setStatus("Upload failed.");
    throw new Error(txt);
  }

  setStatus("Upload complete. Loading proxy…");
  await refreshStatus();
  await loadProxy();
}

async function loadProxy() {
  const st = await getStatus();
  if (!st || !st.proxy_url) {
    setStatus("No proxy URL yet.");
    return;
  }

  VIDEO_EL = $("video");
  if (!VIDEO_EL) return;

  // cache buster
  VIDEO_EL.src = api(st.proxy_url) + `?ts=${Date.now()}`;
  await VIDEO_EL.play().catch(() => {});
  VIDEO_EL.pause();

  setStatus("Proxy loaded. Click player on video 3-5 times, then Save Setup.");
}

async function saveSetup() {
  if (!JOB_ID) throw new Error("No job.");

  const payload = buildSetupPayload();
  if (!payload.clicks || payload.clicks.length < 3) {
    setStatus("Need at least 3 clicks on the player.");
    return;
  }

  setStatus("Saving setup…");
  const res = await fetch(api(`/jobs/${JOB_ID}/setup`), {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!res.ok) {
    const txt = await res.text();
    setStatus("Save setup failed.");
    throw new Error(txt);
  }

  setStatus("Setup saved. Now hit Run.");
  await refreshStatus();
}

async function runJob() {
  if (!JOB_ID) throw new Error("No job.");
  setStatus("Queueing…");

  const res = await fetch(api(`/jobs/${JOB_ID}/run`), { method: "POST" });
  if (!res.ok) {
    const txt = await res.text();
    setStatus("Run failed.");
    throw new Error(txt);
  }

  setStatus("Queued for processing. Tracking…");
  pollUntilDone();
}

async function getStatus() {
  if (!JOB_ID) return null;
  const res = await fetch(api(`/jobs/${JOB_ID}/status`));
  if (!res.ok) return null;
  return await res.json();
}

async function refreshStatus() {
  const st = await getStatus();
  if (!st) return;

  const prog = $("progressBar");
  if (prog && typeof st.progress === "number") prog.value = st.progress;

  const msg = st.message || st.status || "";
  setStatus(msg);
}

async function pollUntilDone() {
  const intervalMs = 1500;

  while (true) {
    await refreshStatus();
    const st = await getStatus();
    if (!st) {
      setStatus("Lost status.");
      return;
    }

    if (st.status === "error") {
      setStatus(st.error || st.message || "Error");
      return;
    }

    if (st.status === "done") {
      setStatus("Done. Loading results…");
      await loadResults();
      return;
    }

    await new Promise(r => setTimeout(r, intervalMs));
  }
}

async function loadResults() {
  if (!JOB_ID) return;

  const res = await fetch(api(`/jobs/${JOB_ID}/results`));
  if (!res.ok) {
    setStatus("No results yet.");
    return;
  }

  const data = await res.json();

  const out = $("results");
  if (out) out.textContent = JSON.stringify(data, null, 2);

  // render links
  const links = $("resultLinks");
  if (links) {
    links.innerHTML = "";
    if (data.combined_url) {
      const a = document.createElement("a");
      a.href = api(data.combined_url);
      a.textContent = "Download combined.mp4";
      a.target = "_blank";
      links.appendChild(a);
      links.appendChild(document.createElement("br"));
    }
    if (Array.isArray(data.clips)) {
      data.clips.forEach((c, i) => {
        const a = document.createElement("a");
        a.href = api(c.url);
        a.textContent = `Clip ${i + 1}: ${c.start.toFixed(2)} → ${c.end.toFixed(2)}`;
        a.target = "_blank";
        links.appendChild(a);
        links.appendChild(document.createElement("br"));
      });
    }
  }

  setStatus("Results ready.");
}

function bindVideoClicks() {
  const v = $("video");
  if (!v) return;

  v.addEventListener("click", (ev) => {
    if (!VIDEO_EL) VIDEO_EL = v;
    const rect = v.getBoundingClientRect();

    const x = (ev.clientX - rect.left) / rect.width;
    const y = (ev.clientY - rect.top) / rect.height;
    const t = v.currentTime || 0;

    addClickSeed(t, x, y);
  });
}

function bindButtons() {
  const btnCreate = $("btnCreateJob");
  const btnUpload = $("btnUpload");
  const btnSave = $("btnSaveSetup");
  const btnRun = $("btnRun");
  const btnClear = $("btnClearSeeds");
  const fileInput = $("videoFile");

  if (btnCreate) btnCreate.onclick = () => createJob().catch(e => setStatus(String(e)));
  if (btnUpload) btnUpload.onclick = () => {
    const f = fileInput && fileInput.files && fileInput.files[0];
    if (!f) { setStatus("Choose a video file first."); return; }
    uploadVideo(f).catch(e => setStatus(String(e)));
  };
  if (btnSave) btnSave.onclick = () => saveSetup().catch(e => setStatus(String(e)));
  if (btnRun) btnRun.onclick = () => runJob().catch(e => setStatus(String(e)));
  if (btnClear) btnClear.onclick = () => clearSeeds();
}

window.addEventListener("DOMContentLoaded", () => {
  bindButtons();
  bindVideoClicks();
  renderSeeds();
  setStatus("Ready. Click Create Job.");
});

