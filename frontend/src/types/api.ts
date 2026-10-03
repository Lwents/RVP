export type VoiceGender = "female" | "male";
export type BgmMode = "demucs" | "ducking" | "none";
export type JobStatus = "queued" | "processing" | "completed" | "failed";
export type ReviewJobStatus = JobStatus | "needs_review" | "ready_to_render";
export type SourceLanguage = "auto" | "en" | "zh" | "vi";
export type ProcessingMode = "fast" | "balanced" | "quality";

export interface CustomBlurBox {
  x_percent: number;
  y_percent: number;
  width_percent: number;
  height_percent: number;
  start_seconds?: number | null;
  end_seconds?: number | null;
}

export interface DubbingRequest {
  source_url?: string | null;
  local_file_path?: string | null;
  processing_mode: ProcessingMode;
  voice_gender: VoiceGender;
  bgm_mode: BgmMode;
  use_demucs: boolean;
  video_speed: number;
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
  completed_at?: string | null;
  output_video_url?: string | null;
  seo_title?: string | null;
  seo_description?: string | null;
  seo_tags?: string[] | null;
  final_evaluation?: ReviewFinalEvaluation | null;
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
  processing_mode: ProcessingMode;
  source_language: SourceLanguage;
  notes?: string | null;
  hard_subtitles: boolean;
  subtitle_x_percent: number;
  subtitle_y_percent: number;
  subtitle_font_size: number;
  subtitle_box_enabled: boolean;
  subtitle_box_opacity: number;
  subtitle_box_height_percent: number;
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
  output_resolution: string;
}

export interface ReviewBeatCandidate {
  candidate_id: string;
  scene_id: string;
  start_seconds: number;
  end_seconds: number;
  thumbnail_url?: string | null;
  match_score?: number | null;
  match_reason?: string | null;
}

export interface ReviewBeat {
  time_hint: string;
  start_seconds?: number | null;
  end_seconds?: number | null;
  purpose: string;
  narration: string;
  segment_id?: string | null;
  event_id?: string | null;
  scene_id?: string | null;
  voice_start?: number | null;
  voice_end?: number | null;
  voice_start_seconds?: number | null;
  voice_end_seconds?: number | null;
  match_score?: number | null;
  match_reason?: string | null;
  selected_candidate_id?: string | null;
  thumbnail_url?: string | null;
  candidates?: ReviewBeatCandidate[];
}

export interface ReviewSegmentPatch {
  narration?: string;
  candidate_id?: string | null;
  scene_id?: string | null;
  start_seconds?: number | null;
  end_seconds?: number | null;
}

export interface ReviewQualityIssue {
  severity: string;
  code: string;
  message: string;
  segment_id?: string | null;
}

export interface ReviewQualityReport {
  phase: string;
  overall_score: number;
  direct_visual_match_percent: number;
  chronology_score: number;
  evidence_score: number;
  character_consistency_score: number;
  duration_adherence_score?: number;
  story_coherence_score?: number;
  style_adherence_score?: number;
  source_coverage_score?: number;
  passed: boolean;
  issues: ReviewQualityIssue[];
}

export type ReviewFinalEvaluationVerdict = "excellent" | "good" | "needs_improvement" | "poor";

export type ReviewFinalEvaluationCriterionKey =
  | "content_fidelity"
  | "translation_accuracy"
  | "av_subtitle_sync"
  | "narrative_coherence"
  | "technical_quality"
  | "safety_compliance";

export interface ReviewFinalEvaluationCriterion {
  key: ReviewFinalEvaluationCriterionKey;
  label: string;
  score: number;
  weight_percent: number;
  passed: boolean;
  feedback: string;
  findings: string[];
}

export interface ReviewFinalEvaluation {
  phase: "final_evaluation";
  overall_score: number;
  verdict: ReviewFinalEvaluationVerdict;
  passed: boolean;
  summary: string;
  strengths: string[];
  recommendations: string[];
  criteria: ReviewFinalEvaluationCriterion[];
  model: string;
  fallback_used: boolean;
  evaluated_at: string;
}

export interface ReviewDraftResult {
  title: string;
  target_minutes: number;
  estimated_duration_seconds?: number | null;
  narration_duration_seconds?: number | null;
  output_duration_seconds?: number | null;
  actual_duration_seconds?: number | null;
  hook: string;
  summary: string;
  narration_script: string;
  beats: ReviewBeat[];
  thumbnail_text: string;
  tags: string[];
  subtitle_file_path?: string | null;
  output_video_url?: string | null;
  output_file_path?: string | null;
  quality_report?: ReviewQualityReport | null;
  final_evaluation?: ReviewFinalEvaluation | null;
}

export interface ReviewDraftJob {
  job_id: string;
  status: ReviewJobStatus;
  progress: number;
  stage: string;
  request: ReviewDraftRequest;
  created_at: string;
  updated_at: string;
  result?: ReviewDraftResult | null;
  error?: string | null;
}
