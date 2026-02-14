// ShiftClipper MVP - Web UI (robust binding)
// Binds buttons even if IDs change by also matching button text.

let JOB_ID = null;
let VIDEO_EL = null;
let CLICK_SEEDS = [];

function $(id) { return document.getElementById(id); }

function api(path) {
  return path.startsWith("/") ? path : `/${path}`;
}

function setStatus(msg) {
  const el = $("statusText") || document.querySelector("[data-status]") || document.querySelector("#status") || document.querySelector(".status");
  if (el) el.textContent = msg;
}

function setJobId(id) {
  JOB_ID = id;
  const el = $("jobIdText") || document.querySelector("[data-jobid]") || document.querySelector("#jobId");
  if (el) el.textContent = id || "(none)";
}

function getElAny(ids) {
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) return el;
  }
  return null;
}

function findButtonByText(needle) {
  const n = needle.toLowerCase();
  const buttons = Array.from(document.querySelectorAll("button, input[type='button'], input[type='submit']"));
  return buttons.find(b => (b.textContent || b.value || "").trim().toLowerCase().includes(n)) || null;
}

function getFormValueAny(ids, fallback = "") {
  const el = getElAny(ids);
  if (!el) return fallback;
  if (el.type === "checkbox") return !!el.checked;
  return el.value ?? fallback;
}

function buildSetupPayload() {
  const camera_mode = getFormValueAny(["cameraMode","camera_mode","camera"], "broadcast");
  const player_number = getFormValueAny(["playerNumber","player_number","number"], "");
  const jersey_color = getFormValueAny(["jerseyColor","jersey_color","color"], "#112d27");
  const extend_sec = parseFloat(getFormValueAny(["extendSec","extend_sec","extend"], "5")) || 5;
  const verify_mode = !!getFormValueAny(["verifyMode","verify_mode","verify"], false);

  const min_clip_len = parseFloat(getFormValueAny(["minClipLen","min_clip_len"], "20")) || 20;
  const pre_roll = parseFloat(getFormValueAny(["preRoll","pre_roll"], "4")) || 4;
  const post_roll = parseFloat(getFormValueAny(["postRoll","post_roll"], "1.5")) || 1.5;
  const sticky_seconds = parseFloat(getFormValueAny(["stickySeconds","sticky_seconds"], "1.5")) || 1.5;
  const detect_stride = parseInt(getFormValueAny(["detectStride","detect_stride"], "3"), 10) || 3;

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

function renderSeeds() {
  const el = $("seedList") || document.querySelector("[data-seedlist]") || document.querySelector("#seeds");
  if (el) {
    el.innerHTML = "";
    CLICK_SEEDS.forEach((c, i) => {
      const li = document.createElement("li");
      li.textContent = `#${i + 1}: t=${c.t.toFixed(3)} x=${c.x.toFixed(3)} y=${c.y.toFixed(3)}`;
      el.appendChild(li);
    });
  }
  const cnt = $("seedCount") || document.querySelector("[data-seedcount]");
  if (cnt) cnt.textContent = String(CLICK_SEEDS.length);
}

function addClickSeed(t, x, y) {
  CLICK_SEEDS.push({ t, x, y });
  renderSeeds();
}

function clearSeeds() {
  CLICK_SEEDS = [];
  renderSeeds();
}

async function createJob() {
  setStatus("Creating job…");
  const res = await fetch(api("/jobs"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: "job" })
  });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  setJobId(data.job_id);
  setStatus("Job created. Upload a video.");
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

  const prog = $("progressBar") || document.querySelector("progress");
  if (prog && typeof st.progress === "number") prog.value = st.progress;

  setStatus(st.message || st.status || "");
}

async function loadProxy() {
  const st = await getStatus();
  if (!st || !st.proxy_url) {
    setStatus("No proxy URL yet.");
    return;
  }
  VIDEO_EL = $("video") || document.querySelector("video");
  if (!VIDEO_EL) {
    setStatus("No <video> element found.");
    return;
  }
  VIDEO_EL.src = api(st.proxy_url) + `?ts=${Date.now()}`;
  await VIDEO_EL.play().catch(() => {});
  VIDEO_EL.pause();
  setStatus("Proxy loaded. Click player 3–5 times, then Save Setup.");
}

async function uploadVideo(file) {
  if (!JOB_ID) throw new Error("No job. Click Create Job first.");
  setStatus("Uploading…");

  const form = new FormData();
  form.append("file", file);

  const res = await fetch(api(`/jobs/${JOB_ID}/upload`), { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());

  await refreshStatus();
  await loadProxy();
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
  if (!res.ok) throw new Error(await res.text());

  setStatus("Setup saved. Now hit Run.");
  await refreshStatus();
}

async function runJob() {
  if (!JOB_ID) throw new Error("No job.");
  setStatus("Queueing…");

  const res = await fetch(api(`/jobs/${JOB_ID}/run`), { method: "POST" });
  if (!res.ok) throw new Error(await res.text());

  setStatus("Queued for processing. Tracking…");
  pollUntilDone();
}

async function loadResults() {
  if (!JOB_ID) return;
  const res = await fetch(api(`/jobs/${JOB_ID}/results`));
  if (!res.ok) {
    setStatus("No results yet.");
    return;
  }
  const data = await res.json();

  const out = $("results") || document.querySelector("pre");
  if (out) out.textContent = JSON.stringify(data, null, 2);

  const links = $("resultLinks") || document.querySelector("[data-resultlinks]");
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

async function pollUntilDone() {
  const intervalMs = 1500;
  while (true) {
    await refreshStatus();
    const st = await getStatus();
    if (!st) { setStatus("Lost status."); return; }
    if (st.status === "error") { setStatus(st.error || st.message || "Error"); return; }
    if (st.status === "done") { setStatus("Done. Loading results…"); await loadResults(); return; }
    await new Promise(r => setTimeout(r, intervalMs));
  }
}

function bindVideoClicks() {
  const v = $("video") || document.querySelector("video");
  if (!v) return;

  v.addEventListener("click", (ev) => {
    VIDEO_EL = v;
    const rect = v.getBoundingClientRect();
    const x = (ev.clientX - rect.left) / rect.width;
    const y = (ev.clientY - rect.top) / rect.height;
    const t = v.currentTime || 0;
    addClickSeed(t, x, y);
  });
}

function bindButtonsRobust() {
  const fileInput =
    $("videoFile") ||
    getElAny(["file","uploadFile"]) ||
    document.querySelector("input[type='file']");

  const btnCreate =
    getElAny(["btnCreateJob","createJob","create_job"]) ||
    findButtonByText("create job") ||
    findButtonByText("create");

  const btnUpload =
    getElAny(["btnUpload","upload","uploadVideo"]) ||
    findButtonByText("upload");

  const btnSave =
    getElAny(["btnSaveSetup","saveSetup","save_setup"]) ||
    findButtonByText("save setup") ||
    findButtonByText("save");

  const btnRun =
    getElAny(["btnRun","run","runJob"]) ||
    findButtonByText("run");

  const btnClear =
    getElAny(["btnClearSeeds","clearSeeds","clear_seeds"]) ||
    findButtonByText("clear");

  if (btnCreate) btnCreate.onclick = () => createJob().catch(e => setStatus(String(e)));
  if (btnUpload) btnUpload.onclick = () => {
    const f = fileInput && fileInput.files && fileInput.files[0];
    if (!f) { setStatus("Choose a video file first."); return; }
    uploadVideo(f).catch(e => setStatus(String(e)));
  };
  if (btnSave) btnSave.onclick = () => saveSetup().catch(e => setStatus(String(e)));
  if (btnRun) btnRun.onclick = () => runJob().catch(e => setStatus(String(e)));
  if (btnClear) btnClear.onclick = () => clearSeeds();

  // If create button wasn't found, tell you immediately
  if (!btnCreate) {
    setStatus("ERROR: Could not find Create Job button (ID mismatch). Open console and list buttons.");
    console.warn("Create Job button not found. Buttons on page:", [...document.querySelectorAll("button")].map(b => ({id:b.id, text:b.textContent.trim()})));
  }
}

window.addEventListener("DOMContentLoaded", () => {
  bindButtonsRobust();
  bindVideoClicks();
  renderSeeds();
  setStatus("Ready. Click Create Job.");
});

