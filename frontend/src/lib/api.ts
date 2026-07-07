import type { DubbingRequest, JobProgress, UploadResponse } from "../types/api";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function uploadWatermark(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_URL}/api/uploads/watermark`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json() as Promise<UploadResponse>;
}

export async function uploadVideo(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_URL}/api/uploads/video`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json() as Promise<UploadResponse>;
}

export async function createJob(payload: DubbingRequest): Promise<{ job_id: string; status: string }> {
  return request("/api/jobs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getJob(jobId: string): Promise<JobProgress> {
  return request(`/api/jobs/${jobId}`);
}

export async function listJobs(): Promise<JobProgress[]> {
  return request("/api/jobs");
}

export async function cancelJob(jobId: string): Promise<{ message: string }> {
  return request(`/api/jobs/${jobId}/cancel`, {
    method: "POST",
  });
}

export async function clearJobs(): Promise<void> {
  const response = await fetch(`${API_URL}/api/jobs`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  await response.json().catch(() => undefined);
}

export function toAbsoluteApiUrl(path: string): string {
  return `${API_URL}${path}`;
}

export async function fetchUrlPreview(url: string): Promise<UploadResponse> {
  const response = await fetch(`${API_URL}/api/uploads/url_preview`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ url }),
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new Error(errorData?.detail || `Lỗi tải preview: ${response.status}`);
  }
  return response.json() as Promise<UploadResponse>;
}

export async function getYoutubeAuthUrl(): Promise<{url: string}> {
  return request("/api/youtube/auth-url");
}

export async function sendYoutubeCallbackCode(code: string): Promise<any> {
  return request("/api/youtube/callback", {
    method: "POST",
    body: JSON.stringify({ code })
  });
}

export async function getYoutubeStats(): Promise<any[]> {
  return request("/api/youtube/stats");
}