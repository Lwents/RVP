import type { DubbingRequest, JobProgress, ReviewDraftJob, ReviewDraftRequest, ReviewSegmentPatch, UploadResponse } from "../types/api";

const ENV_API_URL = import.meta.env.VITE_API_URL?.trim();

function resolveApiUrl(): string {
  if (typeof window === "undefined") {
    return ENV_API_URL || "http://127.0.0.1:8000";
  }

  const hostname = window.location.hostname;
  const isLocalFrontend = hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";

  // Local development should not be blocked by an expired Cloudflare quick tunnel in .env.local.
  if (isLocalFrontend) {
    return "http://127.0.0.1:8000";
  }

  return ENV_API_URL || "http://127.0.0.1:8000";
}

const API_URL = resolveApiUrl();

export interface UploadProgress {
  loaded: number;
  total: number | null;
  percent: number | null;
  bytesPerSecond: number | null;
}

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

export async function uploadVideo(
  file: File,
  onProgress?: (progress: UploadProgress) => void,
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  return new Promise<UploadResponse>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    let startedAt = Date.now();
    let lastLoaded = 0;
    let lastTickAt = startedAt;

    xhr.open("POST", `${API_URL}/api/uploads/video`);
    xhr.responseType = "json";

    xhr.upload.onprogress = (event) => {
      if (!onProgress) return;

      const now = Date.now();
      const elapsedMs = Math.max(now - lastTickAt, 1);
      const deltaLoaded = Math.max(event.loaded - lastLoaded, 0);
      const instantaneousBytesPerSecond = deltaLoaded > 0 ? (deltaLoaded * 1000) / elapsedMs : null;
      const totalElapsedMs = Math.max(now - startedAt, 1);
      const averageBytesPerSecond = event.loaded > 0 ? (event.loaded * 1000) / totalElapsedMs : null;

      onProgress({
        loaded: event.loaded,
        total: event.lengthComputable ? event.total : file.size || null,
        percent: event.lengthComputable && event.total > 0 ? Math.min(100, Math.round((event.loaded / event.total) * 100)) : null,
        bytesPerSecond: instantaneousBytesPerSecond ?? averageBytesPerSecond,
      });

      lastLoaded = event.loaded;
      lastTickAt = now;
    };

    xhr.onerror = () => {
      reject(new Error("Không thể upload video. Vui lòng kiểm tra kết nối."));
    };

    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        const detail = typeof xhr.response === "string"
          ? xhr.response
          : xhr.response?.detail || xhr.responseText || `Request failed with status ${xhr.status}`;
        reject(new Error(detail));
        return;
      }

      onProgress?.({
        loaded: file.size,
        total: file.size,
        percent: 100,
        bytesPerSecond: file.size > 0 ? (file.size * 1000) / Math.max(Date.now() - startedAt, 1) : null,
      });

      const payload = xhr.response ?? JSON.parse(xhr.responseText);
      resolve(payload as UploadResponse);
    };

    xhr.send(formData);
  });
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

export async function generateJobMetadata(jobId: string): Promise<JobProgress> {
  return request(`/api/jobs/${jobId}/metadata`, {
    method: "POST",
  });
}

export async function listJobs(): Promise<JobProgress[]> {
  return request("/api/jobs");
}

export async function cancelJob(jobId: string): Promise<{ message: string }> {
  return request(`/api/jobs/${jobId}/cancel`, {
    method: "POST",
  });
}

export async function clearJobs(): Promise<{ message?: string; warnings?: string }> {
  const response = await fetch(`${API_URL}/api/jobs`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json().catch(() => ({}));
}

export function toAbsoluteApiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  return `${API_URL}${path.startsWith("/") ? path : `/${path}`}`;
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
  kind?: "subtitle" | "logo";
}

export interface CheckedBlurConfig {
  blur_box_enabled: boolean;
  blur_box_y_percent: number;
  blur_box_height_percent: number;
  custom_blur_boxes: Array<{
    x_percent: number;
    y_percent: number;
    width_percent: number;
    height_percent: number;
  }>;
  subtitle_y_percent?: number | null;
}

export async function detectBlurRegions(
  videoPath: string,
  detectSub: boolean = true,
  detectLogo: boolean = true,
  engine: string = "local",
): Promise<{ 
  regions: DetectedBlurRegion[]; 
  count: number; 
  engine_requested: string;
  engine_used: "ai" | "local" | "local_fallback";
  ai_model?: string | null;
  ai_error?: string | null;
  config?: CheckedBlurConfig;
  review?: {
    ok: boolean;
    notes: string[];
  };
  auto_logo?: { 
    watermark_file_name: string; 
    logo_width: number;
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

export async function createReviewDraftJob(payload: ReviewDraftRequest): Promise<ReviewDraftJob> {
  return request("/api/review/jobs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function listReviewDraftJobs(): Promise<ReviewDraftJob[]> {
  return request("/api/review/jobs");
}

export async function getReviewDraftJob(jobId: string): Promise<ReviewDraftJob> {
  return request(`/api/review/jobs/${jobId}`);
}

export async function renderReviewDraftJob(jobId: string): Promise<ReviewDraftJob> {
  return request(`/api/review/jobs/${encodeURIComponent(jobId)}/render`, {
    method: "POST",
  });
}

export async function updateReviewDraftSegment(
  jobId: string,
  segmentId: string,
  payload: ReviewSegmentPatch,
): Promise<ReviewDraftJob> {
  return request(`/api/review/jobs/${encodeURIComponent(jobId)}/segments/${encodeURIComponent(segmentId)}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export async function clearReviewDraftJobs(): Promise<{ message?: string; warnings?: string }> {
  const response = await fetch(`${API_URL}/api/review/jobs`, { method: "DELETE" });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json().catch(() => ({}));
}
