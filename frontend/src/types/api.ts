export type VoiceGender = "female" | "male";
export type BgmMode = "demucs" | "ducking" | "none";
export type LogoPosition = "top_right" | "top_left" | "bottom_right" | "bottom_left";
export type PublishTarget = "youtube" | "facebook";
export type JobStatus = "queued" | "processing" | "completed" | "failed";
export type SourceLanguage = "auto" | "en" | "zh" | "vi";

export interface DubbingRequest {
  source_url?: string | null;
  local_file_path?: string | null;
  voice_gender: VoiceGender;
  bgm_mode: BgmMode;
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
  logo_position: LogoPosition;
  logo_width: number;
  watermark_file_name?: string | null;
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
  error?: string | null;
}

export interface UploadResponse {
  file_name: string;
  content_type?: string | null;
  size: number;
  url: string;
  local_file_path?: string | null;
}
