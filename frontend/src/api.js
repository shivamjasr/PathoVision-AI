const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const TOKEN_KEY = "pathovision_access_token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = getToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });

  if (response.status === 401) {
    clearToken();
  }

  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `Request failed (${response.status}).`);
  }

  return response.json();
}

export async function getHealth() {
  return request("/api/health");
}

export async function login(username, password) {
  const body = new URLSearchParams();
  body.set("username", username);
  body.set("password", password);
  body.set("grant_type", "password");

  const response = await fetch(`${API_BASE}/api/auth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || "Login failed.");
  }

  const data = await response.json();
  setToken(data.access_token);
  return data;
}

export async function getMe() {
  return request("/api/auth/me");
}

export async function analyzeSlide(file, { mode = "classify", tileSize = 256, minTissue = 0.25, idempotencyKey } = {}) {
  const params = new URLSearchParams({
    mode,
    tile_size: String(tileSize),
    min_tissue: String(minTissue),
  });
  const formData = new FormData();
  formData.append("file", file);

  const headers = {};
  if (idempotencyKey) headers["Idempotency-Key"] = idempotencyKey;

  return request(`/api/analyze?${params.toString()}`, {
    method: "POST",
    body: formData,
    headers,
  });
}

export async function getJob(jobId) {
  return request(`/api/jobs/${jobId}`);
}

export async function listJobs(limit = 25) {
  return request(`/api/jobs?limit=${limit}`);
}

export async function getJobEvents(jobId) {
  return request(`/api/jobs/${jobId}/events`);
}

export async function listArtifacts(jobId) {
  return request(`/api/jobs/${jobId}/artifacts`);
}

export function artifactUrl(jobId, filename) {
  return `${API_BASE}/api/jobs/${jobId}/files/${filename}`;
}

export async function fetchArtifact(jobId, filename) {
  const token = getToken();
  const response = await fetch(artifactUrl(jobId, filename), {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!response.ok) {
    throw new Error(`Could not load artifact (${response.status}).`);
  }
  return response.blob();
}
