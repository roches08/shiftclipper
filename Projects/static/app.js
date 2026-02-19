const healthEl = document.getElementById('health');
const statusEl = document.getElementById('status');
const jobIdEl = document.getElementById('jobId');

let currentJobId = null;

function setStatus(data) {
  statusEl.textContent = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
}

function setJobId(jobId) {
  currentJobId = jobId;
  jobIdEl.textContent = jobId || '(none)';
}

async function loadHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    healthEl.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    healthEl.textContent = `Failed to load /api/health: ${err}`;
  }
}

async function createJob() {
  const name = document.getElementById('jobName').value || 'runpod-job';
  const res = await fetch('/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name })
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(JSON.stringify(data));
  }
  setJobId(data.job_id);
  setStatus(data);
}

async function uploadFile() {
  if (!currentJobId) {
    setStatus('Create a job first.');
    return;
  }
  const fileInput = document.getElementById('videoFile');
  const file = fileInput.files?.[0];
  if (!file) {
    setStatus('Choose a video file first.');
    return;
  }

  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`/jobs/${currentJobId}/upload`, {
    method: 'POST',
    body: form
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(JSON.stringify(data));
  }
  setStatus(data);
}

async function runJob() {
  if (!currentJobId) {
    setStatus('Create a job first.');
    return;
  }
  const res = await fetch(`/jobs/${currentJobId}/run`, { method: 'POST' });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(JSON.stringify(data));
  }
  setStatus(data);
}

async function pollStatus() {
  if (!currentJobId) {
    setStatus('Create a job first.');
    return;
  }
  const res = await fetch(`/jobs/${currentJobId}/status`);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(JSON.stringify(data));
  }
  setStatus(data);
}

function wrap(fn) {
  return async () => {
    try {
      await fn();
    } catch (err) {
      setStatus(`Request failed: ${err}`);
    }
  };
}

document.getElementById('createJob').addEventListener('click', wrap(createJob));
document.getElementById('uploadFile').addEventListener('click', wrap(uploadFile));
document.getElementById('runJob').addEventListener('click', wrap(runJob));
document.getElementById('pollStatus').addEventListener('click', wrap(pollStatus));

loadHealth();
