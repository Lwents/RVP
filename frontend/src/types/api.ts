export type VoiceGender = "female" | "male";
export type BgmMode = "demucs" | "ducking" | "none";
export type PublishTarget = "youtube" | "facebook";
export type JobStatus = "queued" | "processing" | "completed" | "failed";
export type SourceLanguage = "auto" | "en" | "zh" | "vi";

export interface CustomBlurBox {
  x_percent: number;
  y_percent: number;
  width_percent: number;
  height_percent: number;
}

export interface DubbingRequest {
  source_url?: string | null;
  local_file_path?: string | null;
  voice_gender: VoiceGender;
  bgm_mode: BgmMode;
  use_demucs: boolean;
  video_speed: number;
  auto_publish: PublishTarget[];
  clone_voice: boolean;
  hard_subtitles: boolean;
  source_has_hard_subtitles: boolean;
  subtitle_x_percent: number;
  subtitle_y_percent: number;
  subtitle_font_size: number;
  subtitle_box_enabled: boolean;
  subtitle_box_opacity: number;
  subtitle_box_height_percent: number;
  source_language: SourceLanguage;
  ducking_volume_db: number;
  output_resolution: string;
  logo_enabled: boolean;
  logo_width: number;
  logo_x_percent: number;
  logo_y_percent: number;
  cinematic_bars_enabled: boolean;
  cinematic_bars_height_percent: number;
  blur_box_enabled: boolean;
  blur_box_y_percent: number;
  blur_box_height_percent: number;
  custom_blur_boxes: CustomBlurBox[];
  watermark_file_name?: string | null;
  auto_detect_sub?: boolean;
  auto_detect_logo?: boolean;
}

export interface JobProgress {
  job_id: string;
  status: JobStatus;
  progress: number;
  stage: string;
  request: DubbingRequest;
  created_at: string;
  updated_at: string;
  output_video_url?: string | null;
  seo_title?: string | null;
  seo_description?: string | null;
  seo_tags?: string[] | null;
  error?: string | null;
}

export interface UploadResponse {
  file_name: string;
  content_type?: string | null;
  size: number;
  url: string;
  local_file_path?: string | null;
}

export interface ReviewDraftRequest {
  video_path: string;
  target_minutes: number;
  style: "story" | "fast" | "emotional" | "funny";
  source_language: SourceLanguage;
  notes?: string | null;
}

export interface ReviewBeat {
  time_hint: string;
  start_seconds?: number | null;
  end_seconds?: number | null;
  purpose: string;
  narration: string;
}

export interface ReviewDraftResult {
  title: string;
  target_minutes: number;
  hook: string;
  summary: string;
  narration_script: string;
  beats: ReviewBeat[];
  thumbnail_text: string;
  tags: string[];
  subtitle_file_path?: string | null;
  output_video_url?: string | null;
  output_file_path?: string | null;
}

export interface ReviewDraftJob {
  job_id: string;
  status: JobStatus;
  progress: number;
  stage: string;
  request: ReviewDraftRequest;
  created_at: string;
  updated_at: string;
  result?: ReviewDraftResult | null;
  error?: string | null;
}
