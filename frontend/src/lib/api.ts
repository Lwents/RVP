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

export async function getYoutubeAuthUrl(redirectUri: string): Promise<{url: string}> {
  return request(`/api/youtube/auth-url?redirect_uri=${encodeURIComponent(redirectUri)}`);
}

export async function sendYoutubeCallbackCode(code: string, redirectUri: string): Promise<any> {
  return request("/api/youtube/callback", {
    method: "POST",
    body: JSON.stringify({ code, redirect_uri: redirectUri })
  });
}

export async function getYoutubeStats(): Promise<any> {
  return request("/api/youtube/stats");
}

export async function uploadYoutubeClientSecret(file: File): Promise<{ status: string; message: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_URL}/api/youtube/client-secret`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const detail = await response.text();
    try {
      const parsed = JSON.parse(detail);
      throw new Error(parsed.detail || "Upload client_secret failed");
    } catch {
      throw new Error(detail || "Upload client_secret failed");
    }
  }
  return response.json() as Promise<{ status: string; message: string }>;
}

export interface DetectedBlurRegion {
  x_percent: number;
  y_percent: number;
  width_percent: number;
  height_percent: number;
  label?: string;
}

export async function detectBlurRegions(
  videoPath: string,
  detectSub: boolean = true,
  detectLogo: boolean = true,
  engine: string = "local",
): Promise<{ 
  regions: DetectedBlurRegion[]; 
  count: number; 
  auto_logo?: { 
    watermark_file_name: string; 
    logo_x_percent: number; 
    logo_y_percent: number; 
    logo_enabled: boolean 
  } 
}> {
  return request("/api/analyze/detect-regions", {
    method: "POST",
    body: JSON.stringify({
      video_path: videoPath,
      detect_sub: detectSub,
      detect_logo: detectLogo,
      engine,
    }),
  });
}