import React, { ChangeEvent, useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, Check, Clapperboard, Clock3, Copy, Download, FileImage, ImageOff, Loader2, Play, Plus, RotateCcw, Save, Sparkles, Trash2, Upload, Video, Wand2, XCircle } from "lucide-react";
import { cancelReviewDraftJob, clearReviewDraftJobs, createReviewDraftJob, detectBlurRegions, getReviewDraftJob, listReviewDraftJobs, optimizeReviewDraftJob, renderReviewDraftJob, renderReviewPreviewJob, retryReviewDraftJob, toAbsoluteApiUrl, updateReviewDraftSegment, uploadVideo, uploadWatermark } from "../lib/api";
import type { UploadProgress } from "../lib/api";
import type { CustomBlurBox, ProcessingMode, ReviewBeat, ReviewBeatCandidate, ReviewDraftJob, ReviewDraftRequest, ReviewQualityIssue, SourceLanguage } from "../types/api";
import { LivePreview } from "./LivePreview";
import { ProcessingModeSelector } from "./ProcessingModeSelector";
import { ReviewFinalEvaluationPanel } from "./ReviewFinalEvaluation";

const reviewStyles: Array<{ label: string; value: ReviewDraftRequest["style"]; description: string }> = [
  { label: "Kể chuyện", value: "story", description: "Đi theo nguyên nhân → diễn biến → kết quả để mạch phim liền lạc." },
  { label: "Nhanh gọn", value: "fast", description: "Câu ngắn, nhịp nhanh và chỉ giữ các mốc quan trọng nhất." },
  { label: "Cảm xúc", value: "emotional", description: "Nhấn vào động cơ, quan hệ và cao trào cảm xúc của nhân vật." },
  { label: "Duyên hài", value: "funny", description: "Kể tự nhiên, dí dỏm nhưng vẫn giữ đúng trình tự câu chuyện." },
];

const MIN_REVIEW_MINUTES = 1;
const MAX_REVIEW_MINUTES = 30;

const sourceLanguages: Array<{ label: string; value: SourceLanguage }> = [
  { label: "Tự nhận diện", value: "auto" },
  { label: "Tiếng Trung", value: "zh" },
  { label: "Tiếng Anh", value: "en" },
  { label: "Tiếng Việt", value: "vi" },
];

type ReviewRenderOptions = Pick<
  ReviewDraftRequest,
  | "hard_subtitles"
  | "subtitle_x_percent"
  | "subtitle_y_percent"
  | "subtitle_font_size"
  | "subtitle_box_enabled"
  | "subtitle_box_opacity"
  | "subtitle_box_height_percent"
  | "logo_enabled"
  | "logo_width"
  | "logo_x_percent"
  | "logo_y_percent"
  | "cinematic_bars_enabled"
  | "cinematic_bars_height_percent"
  | "blur_box_enabled"
  | "blur_box_y_percent"
  | "blur_box_height_percent"
  | "custom_blur_boxes"
  | "watermark_file_name"
  | "output_resolution"
>;

const defaultReviewRenderOptions: ReviewRenderOptions = {
  hard_subtitles: true,
  subtitle_x_percent: 50,
  subtitle_y_percent: 92,
  subtitle_font_size: 64,
  subtitle_box_enabled: false,
  subtitle_box_opacity: 0,
  subtitle_box_height_percent: 18,
  logo_enabled: true,
  logo_width: 86,
  logo_x_percent: 6,
  logo_y_percent: 8,
  cinematic_bars_enabled: false,
  cinematic_bars_height_percent: 10,
  blur_box_enabled: true,
  blur_box_y_percent: 80,
  blur_box_height_percent: 13,
  custom_blur_boxes: [],
  watermark_file_name: null,
  output_resolution: "original",
};

const ACTIVE_REVIEW_JOB_STORAGE_KEY = "auto-translate-ai.activeReviewJobId";

function isRunningReviewJob(job: ReviewDraftJob): boolean {
  return job.status === "queued" || job.status === "processing";
}

function shouldPollReviewJob(job: ReviewDraftJob): boolean {
  return isRunningReviewJob(job) || job.status === "needs_review" || job.status === "ready_to_render";
}

function reviewSnapshotTime(job: ReviewDraftJob | null | undefined): number {
  if (!job) return Number.NaN;
  const time = new Date(job.updated_at).getTime();
  return Number.isFinite(time) ? time : Number.NaN;
}

/** True when `next` is an older snapshot of the job we already hold. */
function isStaleReviewSnapshot(previous: ReviewDraftJob | null | undefined, next: ReviewDraftJob): boolean {
  if (!previous || previous.job_id !== next.job_id) return false;
  const previousTime = reviewSnapshotTime(previous);
  const nextTime = reviewSnapshotTime(next);
  if (Number.isNaN(previousTime) || Number.isNaN(nextTime)) return false;
  return nextTime < previousTime;
}

/** True when `next` is strictly newer than the job we already hold. */
function isNewerReviewSnapshot(previous: ReviewDraftJob | null | undefined, next: ReviewDraftJob): boolean {
  if (!previous || previous.job_id !== next.job_id) return false;
  const previousTime = reviewSnapshotTime(previous);
  const nextTime = reviewSnapshotTime(next);
  if (Number.isNaN(previousTime) || Number.isNaN(nextTime)) return false;
  return nextTime > previousTime;
}

/**
 * Copies text without ever throwing. `navigator.clipboard` is undefined on insecure
 * origins (http:// on a LAN IP), so fall back to the legacy execCommand path.
 */
async function copyTextToClipboard(value: string): Promise<boolean> {
  if (!value) return false;

  try {
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
      return true;
    }
  } catch {
    // Fall through to the legacy path below.
  }

  try {
    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    document.body.removeChild(textarea);
    return copied;
  } catch {
    return false;
  }
}

function pickInitialReviewJob(items: ReviewDraftJob[]): ReviewDraftJob | null {
  if (items.length === 0) return null;
  const storedJobId = typeof window === "undefined" ? null : window.localStorage.getItem(ACTIVE_REVIEW_JOB_STORAGE_KEY);
  return items.find((item) => item.job_id === storedJobId) ?? items.find(isRunningReviewJob) ?? items[0];
}

export const MovieReview = React.memo(function MovieReview() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const previewRef = useRef<HTMLVideoElement | null>(null);
  const logoInputRef = useRef<HTMLInputElement | null>(null);
  const didLoadInitialReviewJobRef = useRef(false);
  const segmentSaveInFlightRef = useRef(false);
  const reviewPollInFlightRef = useRef(false);
  // Remembers the exact error text the status poll last showed, so the next
  // successful tick can clear it without wiping unrelated messages.
  const reviewPollErrorRef = useRef<string | null>(null);
  const reviewHistoryPollInFlightRef = useRef(false);
  const outputCacheIdentityRef = useRef<string | null>(null);
  const [videoPath, setVideoPath] = useState("");
  const [sourceVideoUrl, setSourceVideoUrl] = useState<string | null>(null);
  const [videoName, setVideoName] = useState("Chưa có phim được import");
  const [targetMinutes, setTargetMinutes] = useState(8);
  const [style, setStyle] = useState<ReviewDraftRequest["style"]>("story");
  const [processingMode, setProcessingMode] = useState<ProcessingMode>("balanced");
  const [sourceLanguage, setSourceLanguage] = useState<SourceLanguage>("auto");
  const [notes, setNotes] = useState("");
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const [isUploading, setUploading] = useState(false);
  const [isLogoUploading, setLogoUploading] = useState(false);
  const [isDetectingAI, setDetectingAI] = useState(false);
  const [isCreating, setCreating] = useState(false);
  const [isRetrying, setRetrying] = useState(false);
  const [isCancellingReview, setCancellingReview] = useState(false);
  const [isOptimizingReview, setOptimizingReview] = useState(false);
  const [isPreviewingReview, setPreviewingReview] = useState(false);
  const [isRenderingReview, setRenderingReview] = useState(false);
  const [isClearingHistory, setClearingHistory] = useState(false);
  const [job, setJob] = useState<ReviewDraftJob | null>(null);
  const [reviewJobs, setReviewJobs] = useState<ReviewDraftJob[]>([]);
  const [renderOptions, setRenderOptions] = useState<ReviewRenderOptions>(defaultReviewRenderOptions);
  const [watermarkName, setWatermarkName] = useState("Chưa có logo được tải lên");
  const [previewMode, setPreviewMode] = useState<"source" | "output">("source");
  const [selectedBeatIndex, setSelectedBeatIndex] = useState(0);
  const [narrationDrafts, setNarrationDrafts] = useState<Record<string, string>>({});
  const [savingSegmentId, setSavingSegmentId] = useState<string | null>(null);
  const [outputDurationSeconds, setOutputDurationSeconds] = useState<number | null>(null);
  // Cache-busting token for the rendered output. It must only change when a new file is
  // rendered — keying it off job.updated_at remounted the player on every job update and
  // restarted playback from 0 mid-watch.
  const [outputCacheToken, setOutputCacheToken] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const refreshReviewJobs = useCallback(async () => {
    const items = await listReviewDraftJobs();
    setReviewJobs(items);
    return items;
  }, []);

  const setRenderField = useCallback(<K extends keyof ReviewRenderOptions>(key: K, value: ReviewRenderOptions[K]) => {
    setRenderOptions((current) => ({ ...current, [key]: value }));
  }, []);

  const applyReviewJob = useCallback((item: ReviewDraftJob, resetBeat = true) => {
    setJob(item);
    setVideoPath(item.request.video_path);
    setSourceVideoUrl(sourceUrlFromPath(item.request.video_path));
    setVideoName(fileNameFromPath(item.request.video_path));
    setTargetMinutes(normalizeTargetMinutes(item.request.target_minutes));
    setStyle(item.request.style);
    setProcessingMode(item.request.processing_mode ?? "balanced");
    setSourceLanguage(item.request.source_language ?? "auto");
    setNotes(item.request.notes ?? "");
    setRenderOptions(renderOptionsFromRequest(item.request));
    setWatermarkName(item.request.watermark_file_name ? "Logo đã lưu trong job" : "Chưa có logo được tải lên");
    setPreviewMode(item.result?.output_video_url ? "output" : "source");
    if (resetBeat) {
      setSelectedBeatIndex(0);
      setNarrationDrafts({});
    }
    window.localStorage.setItem(ACTIVE_REVIEW_JOB_STORAGE_KEY, item.job_id);
  }, []);

  const hasRunningReviewJobs = reviewJobs.some(isRunningReviewJob);
  const selectedReviewJobId = job?.job_id ?? null;

  useEffect(() => {
    let cancelled = false;
    refreshReviewJobs().then((items) => {
      if (cancelled || didLoadInitialReviewJobRef.current) return;
      const item = pickInitialReviewJob(items);
      if (!item) return;
      didLoadInitialReviewJobRef.current = true;
      applyReviewJob(item);
    }).catch((error) => {
      setMessage(error instanceof Error ? error.message : "Không thể tải lịch sử review job.");
    });
    return () => {
      cancelled = true;
    };
  }, [applyReviewJob, refreshReviewJobs]);

  useEffect(() => {
    if (!job || !shouldPollReviewJob(job)) return;

    const jobId = job.job_id;
    let cancelled = false;
    const timer = window.setInterval(async () => {
      // A slow request must not stack another one on the next tick.
      if (cancelled || reviewPollInFlightRef.current) return;
      reviewPollInFlightRef.current = true;
      try {
        const nextJob = await getReviewDraftJob(jobId);
        if (cancelled) return;
        if (reviewPollErrorRef.current) {
          const staleError = reviewPollErrorRef.current;
          reviewPollErrorRef.current = null;
          setMessage((current) => (current === staleError ? null : current));
        }
        if (nextJob.job_id !== jobId || isStaleReviewSnapshot(job, nextJob)) return;
        if (nextJob.updated_at !== job.updated_at || nextJob.status !== job.status) {
          const outputBecameAvailable = !job.result?.output_video_url && Boolean(nextJob.result?.output_video_url);
          setJob((previous) => (isStaleReviewSnapshot(previous, nextJob) ? previous : nextJob));
          setReviewJobs((items) => items.map((item) => (
            item.job_id === nextJob.job_id && !isStaleReviewSnapshot(item, nextJob) ? nextJob : item
          )));
          if (outputBecameAvailable) setPreviewMode("output");
        }
      } catch (error) {
        if (!cancelled) {
          const text = error instanceof Error ? error.message : "Không thể lấy trạng thái review job.";
          reviewPollErrorRef.current = text;
          setMessage(text);
        }
      } finally {
        reviewPollInFlightRef.current = false;
      }
    }, isRunningReviewJob(job) ? 1800 : 4000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [job]);

  useEffect(() => {
    if (!hasRunningReviewJobs) return;

    let cancelled = false;
    const refreshRunningHistory = async () => {
      if (cancelled || reviewHistoryPollInFlightRef.current) return;
      reviewHistoryPollInFlightRef.current = true;
      try {
        const items = await refreshReviewJobs();
        if (cancelled) return;

        const selectedJob = selectedReviewJobId ? items.find((item) => item.job_id === selectedReviewJobId) : null;
        if (selectedJob) {
          // Background history polling must not overwrite duration/style/notes
          // that the user is currently preparing for the next review, and must never
          // replace a newer local snapshot (that would revert a just-saved edit and
          // restart the dedicated poll's interval every 2.5s).
          setJob((previous) => (isNewerReviewSnapshot(previous, selectedJob) ? selectedJob : previous));
          return;
        }

        const runningJob = items.find(isRunningReviewJob);
        if (runningJob) {
          applyReviewJob(runningJob);
        }
      } catch {
        // The selected review job poll will surface hard failures; history refresh stays quiet.
      } finally {
        reviewHistoryPollInFlightRef.current = false;
      }
    };

    const timer = window.setInterval(refreshRunningHistory, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [applyReviewJob, hasRunningReviewJobs, refreshReviewJobs, selectedReviewJobId]);

  const openPicker = useCallback(() => {
    if (!isUploading) inputRef.current?.click();
  }, [isUploading]);

  const openLogoPicker = useCallback(() => {
    if (!isLogoUploading) logoInputRef.current?.click();
  }, [isLogoUploading]);

  const handleWatermark = useCallback(async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLogoUploading(true);
    setMessage(null);
    try {
      const uploaded = await uploadWatermark(file);
      setRenderField("watermark_file_name", uploaded.file_name);
      setRenderField("logo_enabled", true);
      setWatermarkName(file.name);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Tải logo thất bại.");
    } finally {
      setLogoUploading(false);
      event.target.value = "";
    }
  }, [setRenderField]);

  const addBlurBox = useCallback(() => {
    setRenderOptions((current) => ({
      ...current,
      custom_blur_boxes: [
        ...current.custom_blur_boxes,
        { x_percent: 72, y_percent: 6, width_percent: 20, height_percent: 9 },
      ],
    }));
  }, []);

  const detectReviewRegions = useCallback(async () => {
    if (!videoPath.trim()) {
      setMessage("Import phim trước khi cho AI nhận diện vùng mờ.");
      return;
    }

    setDetectingAI(true);
    setMessage("AI đang nhận diện logo/sub gốc để đặt vùng làm mờ...");
    try {
      const result = await detectBlurRegions(videoPath, true, true, "ai");
      if (result.config) {
        setRenderOptions((current) => ({
          ...current,
          blur_box_enabled: result.config?.blur_box_enabled ?? current.blur_box_enabled,
          blur_box_y_percent: result.config?.blur_box_y_percent ?? current.blur_box_y_percent,
          blur_box_height_percent: result.config?.blur_box_height_percent ?? current.blur_box_height_percent,
          custom_blur_boxes: result.config?.custom_blur_boxes ?? current.custom_blur_boxes,
          subtitle_y_percent: 92,
        }));
      }
      if (result.auto_logo) {
        setRenderOptions((current) => ({
          ...current,
          watermark_file_name: result.auto_logo?.watermark_file_name ?? current.watermark_file_name,
          logo_width: result.auto_logo?.logo_width ?? current.logo_width,
          logo_x_percent: result.auto_logo?.logo_x_percent ?? current.logo_x_percent,
          logo_y_percent: result.auto_logo?.logo_y_percent ?? current.logo_y_percent,
          logo_enabled: result.auto_logo?.logo_enabled ?? current.logo_enabled,
        }));
        setWatermarkName("logo kenh.png");
      }
      setMessage(result.count > 0 ? `AI đã nhận diện ${result.count} vùng cần xử lý.` : "AI chưa thấy vùng logo/sub rõ ràng, bạn có thể thêm box thủ công.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "AI nhận diện vùng mờ thất bại.");
    } finally {
      setDetectingAI(false);
    }
  }, [videoPath]);

  const handleImport = useCallback(async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setMessage(null);
    setUploadProgress({ loaded: 0, total: file.size || null, percent: 0, bytesPerSecond: null });
    try {
      const uploaded = await uploadVideo(file, setUploadProgress);
      if (!uploaded.local_file_path) {
        throw new Error("Backend không trả về đường dẫn phim đã import.");
      }
      setVideoPath(uploaded.local_file_path);
      setSourceVideoUrl(toAbsoluteApiUrl(uploaded.url));
      setVideoName(file.name);
      setPreviewMode("source");
      setSelectedBeatIndex(0);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Import phim thất bại.");
      setUploadProgress(null);
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }, []);

  const createDraft = useCallback(async () => {
    if (!videoPath.trim()) {
      setMessage("Import phim trước khi tạo review.");
      return;
    }

    setCreating(true);
    setMessage(null);
    try {
      const normalizedTargetMinutes = normalizeTargetMinutes(targetMinutes);
      setTargetMinutes(normalizedTargetMinutes);
      const created = await createReviewDraftJob({
        video_path: videoPath,
        target_minutes: normalizedTargetMinutes,
        style,
        processing_mode: processingMode,
        source_language: sourceLanguage,
        notes: notes.trim() || null,
        ...renderOptions,
      });
      applyReviewJob(created);
      await refreshReviewJobs();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Không thể tạo review job.");
    } finally {
      setCreating(false);
    }
  }, [applyReviewJob, notes, processingMode, refreshReviewJobs, renderOptions, sourceLanguage, style, targetMinutes, videoPath]);

  const retryFailedJob = useCallback(async () => {
    if (!job || job.status !== "failed") return;

    setRetrying(true);
    setMessage(null);
    try {
      const retried = await retryReviewDraftJob(job.job_id, processingMode);
      applyReviewJob(retried);
      await refreshReviewJobs();
      setMessage("Đã chạy lại job từ checkpoint đã lưu.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Không thể chạy lại review job.");
    } finally {
      setRetrying(false);
    }
  }, [applyReviewJob, job, processingMode, refreshReviewJobs]);

  const cancelRunningReview = useCallback(async () => {
    if (!job || !isRunningReviewJob(job) || isCancellingReview) return;
    if (!window.confirm("Hủy job review đang xử lý? Bạn có thể chạy lại từ checkpoint sau đó.")) return;
    setCancellingReview(true);
    setMessage(null);
    try {
      const cancelled = await cancelReviewDraftJob(job.job_id);
      applyReviewJob(cancelled, false);
      await refreshReviewJobs();
      setMessage("Đã hủy job review. Bạn có thể chạy lại từ checkpoint khi sẵn sàng.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Không thể hủy job review.");
    } finally {
      setCancellingReview(false);
    }
  }, [applyReviewJob, isCancellingReview, job, refreshReviewJobs]);

  const selectReviewJob = useCallback((item: ReviewDraftJob) => {
    applyReviewJob(item);
  }, [applyReviewJob]);

  const result = job?.result ?? null;
  const qualityReport = result?.quality_report ?? null;
  const reviewIsBusy = Boolean(job && isRunningReviewJob(job));
  const isSavingAnySegment = savingSegmentId !== null;
  const reviewConfigurationChanged = Boolean(job && (
    videoPath !== job.request.video_path
    || normalizeTargetMinutes(targetMinutes) !== normalizeTargetMinutes(job.request.target_minutes)
    || style !== job.request.style
    || processingMode !== (job.request.processing_mode ?? "balanced")
    || sourceLanguage !== (job.request.source_language ?? "auto")
    || notes.trim() !== (job.request.notes ?? "").trim()
    || JSON.stringify(renderOptions) !== JSON.stringify(renderOptionsFromRequest(job.request))
  ));
  const qualityIssuesBySegment = (qualityReport?.issues ?? []).reduce<Record<string, ReviewQualityIssue[]>>((groups, issue) => {
    if (!issue.segment_id) return groups;
    (groups[issue.segment_id] ??= []).push(issue);
    return groups;
  }, {});
  const canRenderApprovedReview = Boolean(
    job
    && job.status === "ready_to_render"
    && qualityReport?.passed
    && !reviewIsBusy
    && !isRenderingReview
    && !isPreviewingReview
    && !isSavingAnySegment
    && !reviewConfigurationChanged,
  );
  const canAutoOptimizeReview = Boolean(
    job
    && result
    && !reviewIsBusy
    && !isSavingAnySegment
    && !isRenderingReview
    && !isPreviewingReview
    && !isOptimizingReview,
  );
  const canRenderPreviewReview = Boolean(
    job
    && result
    && !reviewIsBusy
    && !isSavingAnySegment
    && !isRenderingReview
    && !isPreviewingReview
    && !isOptimizingReview
    && !reviewConfigurationChanged,
  );
  const renderButtonTitle = canRenderApprovedReview
    ? "Render video từ kịch bản và cảnh đã duyệt"
    : reviewConfigurationChanged
      ? "Cấu hình bên trái đã thay đổi. Hãy tạo lại bản review để áp dụng đúng thời lượng, phong cách và tùy chọn render."
      : reviewIsBusy || isRenderingReview
      ? "Job đang xử lý; vui lòng đợi hoàn tất"
      : qualityReport?.phase === "post_render"
        ? "Hậu kiểm sau render chưa đạt. Mở các segment viền đỏ, sửa lời dẫn hoặc chọn cảnh phù hợp rồi lưu; lỗi không gắn segment được liệt kê trong bảng QA."
        : "QA trước render chưa đạt. Mở các segment viền đỏ, sửa lời dẫn hoặc chọn cảnh phù hợp rồi lưu.";
  const outputVideoBaseUrl = result?.output_video_url ? toAbsoluteApiUrl(result.output_video_url) : null;
  // Identity of the rendered file itself. The backend clears output_video_url while a new
  // render runs, so this flips to null and back whenever a genuinely new file is produced.
  const outputIdentityKey = result?.output_video_url
    ? `${result.output_video_url}::${result.output_file_path ?? ""}`
    : null;
  const outputVideoUrl = outputVideoBaseUrl
    ? outputCacheToken
      ? `${outputVideoBaseUrl}${outputVideoBaseUrl.includes("?") ? "&" : "?"}v=${encodeURIComponent(outputCacheToken)}`
      : outputVideoBaseUrl
    : null;
  const targetDurationSeconds = job ? normalizeTargetMinutes(job.request.target_minutes) * 60 : null;
  const plannedDurationSeconds = reviewPlannedDurationSeconds(result);
  const reportedOutputDurationSeconds = firstFinitePositive([
    result?.output_duration_seconds,
  ]);
  const actualOutputDurationSeconds = outputVideoUrl
    ? reportedOutputDurationSeconds ?? outputDurationSeconds
    : null;
  const durationToAssess = actualOutputDurationSeconds ?? plannedDurationSeconds;
  const durationAssessment = assessReviewDuration(targetDurationSeconds, durationToAssess);
  const selectedStyle = reviewStyles.find((item) => item.value === style) ?? reviewStyles[0];
  const jobStyle = job
    ? reviewStyles.find((item) => item.value === job.request.style) ?? reviewStyles[0]
    : selectedStyle;
  const activePreviewUrl = previewMode === "output" ? outputVideoUrl : sourceVideoUrl;
  const selectedBeat = result?.beats[selectedBeatIndex] ?? null;

  useEffect(() => {
    // The ref keeps StrictMode's double-invoked mount from minting two tokens.
    if (outputCacheIdentityRef.current === outputIdentityKey) return;
    outputCacheIdentityRef.current = outputIdentityKey;
    setOutputCacheToken(outputIdentityKey ? String(Date.now()) : null);
  }, [outputIdentityKey]);

  useEffect(() => {
    setOutputDurationSeconds(null);
  }, [outputVideoUrl]);

  const rememberOutputDuration = useCallback((duration: number) => {
    setOutputDurationSeconds(Number.isFinite(duration) && duration > 0 ? duration : null);
  }, []);

  const seekSourcePreview = useCallback((seconds: number) => {
    const el = previewRef.current;
    if (!el) return;
    const apply = () => {
      el.currentTime = Math.max(0, seconds);
      el.play().catch(() => undefined);
    };
    // Before metadata is loaded, setting currentTime is ignored by some browsers.
    if (el.readyState >= 1) apply();
    else el.addEventListener("loadedmetadata", apply, { once: true });
  }, []);

  // Seek requested while the preview was still on the output tab; applied once the
  // source <video> has actually mounted (a fixed timeout can fire before commit).
  const pendingSourceSeekRef = useRef<number | null>(null);

  useEffect(() => {
    if (previewMode !== "source" || pendingSourceSeekRef.current == null) return;
    const seconds = pendingSourceSeekRef.current;
    pendingSourceSeekRef.current = null;
    seekSourcePreview(seconds);
  }, [previewMode, seekSourcePreview]);

  const jumpToBeat = useCallback((index: number) => {
    const beat = result?.beats[index];
    if (!beat) return;
    setSelectedBeatIndex(index);
    if (beat.start_seconds == null) return;
    if (previewMode !== "source") {
      pendingSourceSeekRef.current = Math.max(0, beat.start_seconds);
      setPreviewMode("source");
      return;
    }
    seekSourcePreview(beat.start_seconds);
  }, [previewMode, result, seekSourcePreview]);

  const previewCandidate = useCallback((index: number, candidate: ReviewBeatCandidate) => {
    setSelectedBeatIndex(index);
    if (previewMode !== "source") {
      pendingSourceSeekRef.current = Math.max(0, candidate.start_seconds);
      setPreviewMode("source");
      return;
    }
    seekSourcePreview(candidate.start_seconds);
  }, [previewMode, seekSourcePreview]);

  const persistSegmentEdit = useCallback(async (
    beat: ReviewBeat,
    index: number,
    candidate?: ReviewBeatCandidate,
  ) => {
    if (!job || !beat.segment_id) {
      setMessage("Job cũ chưa có mã segment nên chỉ có thể xem, chưa thể chỉnh từng câu.");
      return;
    }
    if (segmentSaveInFlightRef.current || isRenderingReview) {
      setMessage("Đang lưu một câu khác; vui lòng đợi hoàn tất để tránh ghi đè lựa chọn.");
      return;
    }

    const draftKey = reviewBeatKey(beat, index);
    const narration = (narrationDrafts[draftKey] ?? beat.narration).trim();
    const narrationChanged = narration !== beat.narration.trim();
    if (!candidate && !narrationChanged) {
      setMessage("Lời dẫn chưa thay đổi nên không cần lưu lại.");
      return;
    }
    segmentSaveInFlightRef.current = true;
    setSavingSegmentId(beat.segment_id);
    setMessage(null);
    try {
      const nextJob = await updateReviewDraftSegment(job.job_id, beat.segment_id, {
        ...(narrationChanged ? { narration } : {}),
        ...(candidate ? {
          candidate_id: candidate.candidate_id,
          scene_id: candidate.scene_id,
          start_seconds: candidate.start_seconds,
          end_seconds: candidate.end_seconds,
        } : {}),
      });
      setJob(nextJob);
      setReviewJobs((items) => items.map((item) => item.job_id === nextJob.job_id ? nextJob : item));
      setNarrationDrafts((current) => ({ ...current, [draftKey]: narration }));
      setSelectedBeatIndex(index);
      setMessage(candidate ? "Đã đổi cảnh nguồn cho câu này." : "Đã lưu lời dẫn của segment.");
    } catch (error) {
      try {
        const refreshed = await getReviewDraftJob(job.job_id);
        setJob(refreshed);
        setReviewJobs((items) => items.map((item) => item.job_id === refreshed.job_id ? refreshed : item));
      } catch {
        // Periodic polling remains the final recovery path.
      }
      setMessage(error instanceof Error ? error.message : "Không thể lưu thay đổi cho segment.");
    } finally {
      segmentSaveInFlightRef.current = false;
      setSavingSegmentId(null);
    }
  }, [isRenderingReview, job, narrationDrafts]);

  const renderApprovedReview = useCallback(async () => {
    if (!job || !canRenderApprovedReview || segmentSaveInFlightRef.current) return;

    setRenderingReview(true);
    setMessage(null);
    try {
      const nextJob = await renderReviewDraftJob(job.job_id);
      setJob(nextJob);
      setReviewJobs((items) => items.map((item) => item.job_id === nextJob.job_id ? nextJob : item));
      setPreviewMode(nextJob.result?.output_video_url ? "output" : "source");
      setMessage("Đã nhận lệnh render video từ bản review đã duyệt.");
    } catch (error) {
      try {
        const refreshed = await getReviewDraftJob(job.job_id);
        setJob(refreshed);
        setReviewJobs((items) => items.map((item) => item.job_id === refreshed.job_id ? refreshed : item));
      } catch {
        // The regular job poll remains the final recovery path.
      }
      setMessage(error instanceof Error ? error.message : "Không thể bắt đầu render video đã duyệt.");
    } finally {
      setRenderingReview(false);
    }
  }, [canRenderApprovedReview, job]);

  const optimizeReviewDraft = useCallback(async () => {
    if (!job || !canAutoOptimizeReview) return;
    setOptimizingReview(true);
    setMessage(null);
    try {
      const nextJob = await optimizeReviewDraftJob(job.job_id);
      setJob(nextJob);
      setReviewJobs((items) => items.map((item) => item.job_id === nextJob.job_id ? nextJob : item));
      setMessage(nextJob.result?.quality_report?.passed
        ? "AI đã tự tối ưu timeline, cảnh và thời lượng; bản nháp đã sẵn sàng render."
        : "AI đã tự xử lý các lỗi nhẹ. Các lỗi nghiêm trọng còn lại được giữ để bạn duyệt.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Không thể tự tối ưu bản nháp.");
    } finally {
      setOptimizingReview(false);
    }
  }, [canAutoOptimizeReview, job]);

  const renderPreviewReview = useCallback(async () => {
    if (!job || !canRenderPreviewReview) return;
    setPreviewingReview(true);
    setMessage(null);
    try {
      const nextJob = await renderReviewPreviewJob(job.job_id);
      setJob(nextJob);
      setReviewJobs((items) => items.map((item) => item.job_id === nextJob.job_id ? nextJob : item));
      setMessage("Đang render bản xem trước. Bản này có thể còn cảnh báo QA.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Không thể render bản xem trước.");
    } finally {
      setPreviewingReview(false);
    }
  }, [canRenderPreviewReview, job]);

  const clearReviewHistory = useCallback(async () => {
    const confirmed = window.confirm("Đưa toàn bộ file review job vào Thùng rác và xóa lịch sử review?");
    if (!confirmed) return;

    setClearingHistory(true);
    setMessage(null);
    try {
      const result = await clearReviewDraftJobs();
      setReviewJobs([]);
      setJob(null);
      setSelectedBeatIndex(0);
      window.localStorage.removeItem(ACTIVE_REVIEW_JOB_STORAGE_KEY);
      setMessage(result.message || "Đã xóa lịch sử review và đưa file vào Thùng rác.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Không thể xóa lịch sử review.");
    } finally {
      setClearingHistory(false);
    }
  }, []);

  const copyText = useCallback(async (value: string) => {
    if (!value) return;
    const copied = await copyTextToClipboard(value);
    setMessage(
      copied
        ? "Đã copy vào clipboard."
        : "Không thể copy tự động (trình duyệt chặn clipboard). Hãy bôi đen nội dung và nhấn Ctrl+C.",
    );
  }, []);

  return (
    <div className="ios-view-transition">
      <header className="page-heading">
        <div>
          <h1>Tạo review phim</h1>
          <p>Nạp phim dài, AI tạo kịch bản kể chuyện và danh sách cảnh gợi ý để dựng review.</p>
        </div>
        <button className="ios-button" type="button" onClick={createDraft} disabled={isUploading || isCreating || hasRunningReviewJobs || !videoPath}>
          {isCreating || hasRunningReviewJobs ? <Loader2 className="spin" size={18} /> : <Sparkles size={18} />}
          {reviewConfigurationChanged ? "Tạo lại theo cấu hình mới" : "Tạo bản review"}
        </button>
      </header>

      <div className="review-workspace">
        <section className="ios-card">
          <div className="section-title">
            <span>01</span>
            <h2>Nguồn phim</h2>
          </div>

          <div className="field">
            <span>File phim</span>
            <div className="upload-row">
              <button className="ios-button" type="button" onClick={openPicker} disabled={isUploading}>
                {isUploading ? <Loader2 className="spin" size={17} /> : <Upload size={17} />}
                <span>{isUploading ? "Đang import..." : "Import phim"}</span>
              </button>
              <input
                ref={inputRef}
                className="file-input-hidden"
                type="file"
                accept="video/*,.mp4,.mov,.mkv,.m4v,.webm"
                onChange={handleImport}
              />
              <span className="upload-name">
                <Video size={17} />
                {videoName}
              </span>
            </div>
            {uploadProgress && (
              <div className="upload-progress">
                <div className="upload-progress-meta">
                  <strong>{isUploading ? `Đang import ${uploadProgress.percent ?? 0}%` : "Import hoàn tất"}</strong>
                  <span>
                    {formatBytes(uploadProgress.loaded)}
                    {uploadProgress.total ? ` / ${formatBytes(uploadProgress.total)}` : ""}
                    {uploadProgress.bytesPerSecond ? ` • ${formatBytes(uploadProgress.bytesPerSecond)}/s` : ""}
                  </span>
                </div>
                <div className="upload-progress-track">
                  <div className="upload-progress-bar" style={{ width: `${uploadProgress.percent ?? 0}%` }} />
                </div>
              </div>
            )}
          </div>

          <div className="two-fields">
            <label className="field">
              <span>Thời lượng review (phút)</span>
              <span className="review-duration-input">
                <input
                  className="ios-input"
                  type="number"
                  inputMode="numeric"
                  min={MIN_REVIEW_MINUTES}
                  max={MAX_REVIEW_MINUTES}
                  step={1}
                  value={targetMinutes}
                  onChange={(event) => {
                    const nextValue = event.currentTarget.valueAsNumber;
                    if (Number.isFinite(nextValue)) setTargetMinutes(nextValue);
                  }}
                  onBlur={() => setTargetMinutes((current) => normalizeTargetMinutes(current))}
                  aria-describedby="review-duration-help"
                />
                <span className="review-input-suffix" aria-hidden="true">phút</span>
              </span>
              <small id="review-duration-help" className="review-field-help">Cho phép {MIN_REVIEW_MINUTES}–{MAX_REVIEW_MINUTES} phút; AI phải viết đủ lời đọc cho mốc này.</small>
            </label>
            <label className="field">
              <span>Ngôn ngữ phim</span>
              <select className="ios-input" value={sourceLanguage} onChange={(event) => setSourceLanguage(event.target.value as SourceLanguage)}>
                {sourceLanguages.map((item) => (
                  <option key={item.value} value={item.value}>{item.label}</option>
                ))}
              </select>
            </label>
          </div>

          <ProcessingModeSelector value={processingMode} onChange={setProcessingMode} />

          <div className="segment-wrap">
            <span className="label">Phong cách review</span>
            <div className="segments">
              {reviewStyles.map((item) => (
                <button
                  key={item.value}
                  type="button"
                  className={style === item.value ? "active" : ""}
                  onClick={() => setStyle(item.value)}
                  aria-pressed={style === item.value}
                  title={item.description}
                >
                  {item.label}
                </button>
              ))}
            </div>
            <small className="review-field-help"><strong>{selectedStyle.label}:</strong> {selectedStyle.description}</small>
          </div>

          <label className="field">
            <span>Ghi chú thêm</span>
            <textarea
              className="ios-input review-notes"
              value={notes}
              onChange={(event) => setNotes(event.target.value)}
              placeholder="Ví dụ: kể kiểu cuốn, tránh spoil kết quá sớm, nhấn mạnh hy sinh của nhân vật chính..."
            />
          </label>

          <ReviewDisplayControls
            options={renderOptions}
            watermarkName={watermarkName}
            isLogoUploading={isLogoUploading}
            isDetectingAI={isDetectingAI}
            onFieldChange={setRenderField}
            onOpenLogoPicker={openLogoPicker}
            onDetectAI={detectReviewRegions}
            onAddBlurBox={addBlurBox}
          />
          <input
            ref={logoInputRef}
            className="file-input-hidden"
            type="file"
            accept="image/*"
            onChange={handleWatermark}
          />

          {message && <div className="review-error">{message}</div>}

          <div className="review-block" style={{ marginTop: "18px" }}>
            <div className="review-block-head">
              <h3>Lịch sử review job</h3>
              {reviewJobs.length > 0 && (
                <button className="ios-button ios-button-secondary" type="button" onClick={clearReviewHistory} disabled={isClearingHistory}>
                  {isClearingHistory ? <Loader2 className="spin" size={15} /> : <Trash2 size={15} />}
                  Xóa tất cả
                </button>
              )}
            </div>
            {reviewJobs.length === 0 ? (
              <p style={{ color: "#86868b", margin: 0 }}>Chưa có review job nào.</p>
            ) : (
              <div className="review-beats">
                {reviewJobs.map((item) => (
                  <button
                    key={item.job_id}
                    type="button"
                    className={`review-beat ${item.job_id === selectedReviewJobId ? "active" : ""}`}
                    aria-current={item.job_id === selectedReviewJobId ? "true" : undefined}
                    onClick={() => selectReviewJob(item)}
                    style={{ textAlign: "left", cursor: "pointer" }}
                  >
                    <strong>{item.stage}</strong>
                    <span>{item.progress}% • {item.status} • {formatDateTime(item.created_at)}</span>
                    <p>{fileNameFromPath(item.request.video_path)} • Mục tiêu {item.request.target_minutes} phút • {reviewStyleName(item.request.style)}</p>
                  </button>
                ))}
              </div>
            )}
          </div>
        </section>

        <div className="review-preview-stack">
          <ReviewLivePreview
            ref={previewRef}
            mode={previewMode}
            onModeChange={setPreviewMode}
            sourceVideoUrl={sourceVideoUrl}
            outputVideoUrl={outputVideoUrl}
            activeVideoUrl={activePreviewUrl}
            selectedBeat={selectedBeat}
            outputDurationSeconds={actualOutputDurationSeconds}
            renderOptions={renderOptions}
            onRenderFieldChange={setRenderField}
            onOutputDurationChange={rememberOutputDuration}
          />

          <section className="ios-card">
          <div className="section-title">
            <span>02</span>
            <h2>Bản review AI</h2>
          </div>

          {!job && (
            <div className="review-empty">
              <Clapperboard size={36} />
              <p>Chưa có bản review nào. Import phim rồi bấm tạo để AI bắt đầu phân tích.</p>
            </div>
          )}

          {job && (
            <div className="review-status">
              <div className="review-status-head">
                <strong>{job.stage}</strong>
                <span>{job.progress}%</span>
              </div>
              <div className="upload-progress-track">
                <div className="upload-progress-bar" style={{ width: `${job.progress}%` }} />
              </div>
              <div className="review-job-spec" aria-label="Cấu hình của job đang xem">
                <span><Clock3 size={14} />Mục tiêu <strong>{job.request.target_minutes} phút</strong></span>
                <span>Phong cách <strong>{jobStyle.label}</strong></span>
                <span>Chế độ <strong>{processingModeLabel(job.request.processing_mode)}</strong></span>
              </div>
              {reviewIsBusy && (
                <button
                  className="ios-button ios-button-secondary review-cancel-button"
                  type="button"
                  onClick={cancelRunningReview}
                  disabled={isCancellingReview}
                  title="Dừng job đang xử lý; sau đó có thể chạy lại từ checkpoint"
                >
                  {isCancellingReview ? <Loader2 className="spin" size={16} /> : <XCircle size={16} />}
                  {isCancellingReview ? "Đang hủy..." : "Hủy xử lý"}
                </button>
              )}
              {reviewConfigurationChanged && (
                <div className="review-duration-warning" role="alert">
                  <AlertTriangle size={17} />
                  <span>Bạn đã đổi thời lượng, phong cách hoặc tùy chọn. Bấm <strong>Tạo lại theo cấu hình mới</strong> trước khi render để các lựa chọn này thực sự được áp dụng.</span>
                </div>
              )}
              {job.error && <div className="review-error">{job.error}</div>}
              {job.status === "failed" && (
                <button
                  className="ios-button ios-button-secondary review-retry-button"
                  type="button"
                  onClick={retryFailedJob}
                  disabled={isRetrying}
                >
                  {isRetrying ? <Loader2 className="spin" size={16} /> : <RotateCcw size={16} />}
                  {isRetrying ? "Đang chạy lại..." : "Chạy lại từ checkpoint"}
                </button>
              )}
            </div>
          )}

          {result && (
            <div className="review-result">
              <section className={`review-duration-summary ${durationAssessment?.significant ? "warning" : ""}`}>
                <div className="review-duration-head">
                  <span><Clock3 size={17} />Đối chiếu thời lượng</span>
                  <strong>Mục tiêu {job?.request.target_minutes ?? result.target_minutes} phút</strong>
                </div>
                <div className="review-duration-facts">
                  {plannedDurationSeconds != null && <span>Voice dự kiến <strong>{formatReviewDuration(plannedDurationSeconds)}</strong></span>}
                  {actualOutputDurationSeconds != null && <span>Video đã render <strong>{formatReviewDuration(actualOutputDurationSeconds)}</strong></span>}
                  <span>Phong cách <strong>{jobStyle.label}</strong></span>
                </div>
                {durationAssessment?.significant && (
                  <div className="review-duration-warning" role="alert">
                    <AlertTriangle size={17} />
                    <span>
                      Bản {actualOutputDurationSeconds != null ? "render" : "nháp"} đang {durationAssessment.direction === "short" ? "ngắn hơn" : "dài hơn"} mục tiêu {formatReviewDuration(durationAssessment.absoluteDeltaSeconds)}
                      {durationAssessment.direction === "short" ? `, mới đạt ${durationAssessment.percentOfTarget}% thời lượng yêu cầu.` : "."}
                      {actualOutputDurationSeconds == null && " Nên tạo lại hoặc bổ sung lời dẫn trước khi render."}
                    </span>
                  </div>
                )}
              </section>
              {qualityReport && (
                <section className={`review-qa-panel ${qualityReport.passed ? "passed" : "needs-review"}`}>
                  <div className="review-qa-head">
                    <div>
                      <span>{qualityReport.phase === "post_render" ? "Hậu kiểm kỹ thuật trên video đã render" : "Kiểm tra chất lượng trước render"}</span>
                      <strong>
                        {qualityReport.phase === "post_render"
                          ? qualityReport.passed ? "Video cuối đã đạt QA kỹ thuật" : "Video cuối còn lỗi cần kiểm tra lại"
                          : qualityReport.passed ? "Đạt QA — sẵn sàng render" : "Cần duyệt lại trước khi render"}
                      </strong>
                    </div>
                    <span className="review-qa-score">{formatQaScore(qualityReport.overall_score)}<small>/100</small></span>
                  </div>
                  <div className="review-qa-metrics">
                    <ReviewQaMetric label="Hình khớp lời" value={qualityReport.direct_visual_match_percent} />
                    <ReviewQaMetric label="Đúng trình tự" value={qualityReport.chronology_score} />
                    <ReviewQaMetric label="Đủ bằng chứng" value={qualityReport.evidence_score} />
                    <ReviewQaMetric label="Nhất quán nhân vật" value={qualityReport.character_consistency_score} />
                    {qualityReport.duration_adherence_score != null && <ReviewQaMetric label="Đúng thời lượng" value={qualityReport.duration_adherence_score} />}
                    {qualityReport.story_coherence_score != null && <ReviewQaMetric label="Mạch truyện" value={qualityReport.story_coherence_score} />}
                    {qualityReport.style_adherence_score != null && <ReviewQaMetric label="Đúng phong cách" value={qualityReport.style_adherence_score} />}
                    {qualityReport.source_coverage_score != null && <ReviewQaMetric label="Phủ timeline" value={qualityReport.source_coverage_score} />}
                  </div>
                  {qualityReport.issues.length > 0 && (
                    <div className="review-qa-issues">
                      {qualityReport.issues.map((issue, issueIndex) => (
                        <span key={`${issue.code}-${issue.segment_id || issueIndex}`}>
                          <AlertTriangle size={14} />
                          {issue.segment_id ? `${issue.segment_id}: ` : ""}{issue.message}
                        </span>
                      ))}
                    </div>
                  )}
                </section>
              )}
              {result.final_evaluation && <ReviewFinalEvaluationPanel evaluation={result.final_evaluation} />}
              <button
                className="ios-button review-render-approved"
                type="button"
                onClick={renderApprovedReview}
                disabled={!canRenderApprovedReview}
                title={renderButtonTitle}
              >
                {reviewIsBusy || isRenderingReview ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
                {reviewIsBusy || isRenderingReview ? "Đang xử lý / render video..." : "Render video đã duyệt"}
              </button>
              <div className="review-automation-actions">
                <button
                  className="ios-button review-auto-optimize"
                  type="button"
                  onClick={optimizeReviewDraft}
                  disabled={!canAutoOptimizeReview}
                  title="Tự rút/gom lời dẫn, chọn cảnh khớp hơn và giữ timeline tăng dần mà không gọi lại AI phân tích toàn bộ phim"
                >
                  {isOptimizingReview ? <Loader2 className="spin" size={18} /> : <Wand2 size={18} />}
                  {isOptimizingReview ? "Đang tự tối ưu..." : "AI tự tối ưu bản nháp"}
                </button>
                <button
                  className="ios-button ios-button-secondary review-render-preview"
                  type="button"
                  onClick={renderPreviewReview}
                  disabled={!canRenderPreviewReview}
                  title="Xuất video để kiểm tra ngay cả khi QA còn cảnh báo không nghiêm trọng"
                >
                  {isPreviewingReview ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
                  {isPreviewingReview ? "Đang render xem trước..." : "Render bản nháp để xem trước"}
                </button>
              </div>
              {result.output_video_url && (
                <a className="ios-button review-download" href={outputVideoUrl ?? toAbsoluteApiUrl(result.output_video_url)} target="_blank" rel="noreferrer">
                  <Download size={17} />
                  {actualOutputDurationSeconds != null
                    ? `Tải video review • ${formatReviewDuration(actualOutputDurationSeconds)}`
                    : `Tải video review • mục tiêu ${job?.request.target_minutes ?? result.target_minutes} phút`}
                </a>
              )}
              <ReviewBlock title="Tiêu đề" value={result.title} onCopy={copyText} />
              <ReviewBlock title="Hook mở đầu" value={result.hook} onCopy={copyText} />
              <ReviewBlock title="Tóm tắt lõi truyện" value={result.summary} onCopy={copyText} />
              <ReviewBlock title="Kịch bản đọc voice" value={result.narration_script} onCopy={copyText} large />
              <div className="review-block">
                <div className="review-block-head">
                  <h3>Danh sách cảnh gợi ý</h3>
                </div>
                <div className="review-beats">
                  {result.beats.map((beat, index) => {
                    const draftKey = reviewBeatKey(beat, index);
                    const score = normalizedMatchScore(beat.match_score);
                    const isLowConfidence = score != null && score < 0.75;
                    const segmentQualityIssues = beat.segment_id ? qualityIssuesBySegment[beat.segment_id] ?? [] : [];
                    const hasSegmentQualityIssue = segmentQualityIssues.length > 0;
                    const needsReview = isLowConfidence || hasSegmentQualityIssue;
                    const candidates = (beat.candidates ?? []).slice(0, 3);
                    const narrationDraft = narrationDrafts[draftKey] ?? beat.narration;
                    const isSaving = Boolean(beat.segment_id && savingSegmentId === beat.segment_id);
                    const narrationDirty = narrationDraft.trim() !== beat.narration.trim();
                    const bestCandidateScore = Math.max(
                      ...candidates.map((candidate) => normalizedMatchScore(candidate.match_score) ?? 0),
                      0,
                    );
                    const thumbnailUrl = beat.thumbnail_url ?? candidates[0]?.thumbnail_url ?? null;

                    return (
                      <article
                        key={beat.segment_id || `${beat.time_hint}-${index}`}
                        className={`review-inspector-card ${index === selectedBeatIndex ? "active" : ""} ${needsReview ? "low-confidence" : ""}`}
                      >
                        <button className="review-inspector-select" type="button" onClick={() => jumpToBeat(index)}>
                          <span className="review-scene-thumbnail">
                            {thumbnailUrl ? (
                              <img src={toAbsoluteApiUrl(thumbnailUrl)} alt={`Khung hình nguồn ${beat.scene_id || index + 1}`} loading="lazy" />
                            ) : (
                              <span className="review-thumbnail-empty"><ImageOff size={22} />Chưa có ảnh</span>
                            )}
                          </span>
                          <span className="review-inspector-summary">
                            <span className="review-inspector-title-row">
                              <strong>{beat.segment_id || `Câu ${index + 1}`}</strong>
                              {(score != null || hasSegmentQualityIssue) && (
                                <span className={`review-match-score ${needsReview ? "low" : ""}`}>
                                  {hasSegmentQualityIssue && qualityReport?.phase === "post_render"
                                    ? "Lỗi hậu kiểm"
                                    : score != null
                                      ? `${Math.round(score * 100)}% khớp`
                                      : "Cần kiểm tra"}
                                </span>
                              )}
                            </span>
                            <span className="review-source-time"><Clock3 size={14} />Nguồn {formatReviewRange(beat.start_seconds, beat.end_seconds, beat.time_hint)}</span>
                            {(beat.voice_start ?? beat.voice_start_seconds) != null && (
                              <span className="review-voice-time">
                                Voice {formatReviewRange(
                                  beat.voice_start ?? beat.voice_start_seconds,
                                  beat.voice_end ?? beat.voice_end_seconds,
                                )}
                              </span>
                            )}
                            <span className="review-purpose">{beat.event_id ? `${beat.event_id} • ` : ""}{beat.purpose}</span>
                            {beat.match_reason && <span className="review-match-reason">{beat.match_reason}</span>}
                          </span>
                        </button>

                        {needsReview && (
                          <div className="review-match-warning">
                            <AlertTriangle size={16} />
                            <span>
                              {hasSegmentQualityIssue
                                ? segmentQualityIssues.map((issue) => issue.message).join(" • ")
                                : bestCandidateScore < 0.75
                                  ? "Không có cảnh thay thế nào đạt 75% — hãy sửa lời dẫn ngắn gọn để chỉ mô tả đúng cảnh đã chọn rồi lưu."
                                  : "Điểm khớp dưới 75% — hãy chọn phương án có điểm cao hơn."}
                            </span>
                          </div>
                        )}

                        <label className="review-narration-editor">
                          <span>Lời dẫn cho câu này</span>
                          <textarea
                            className="ios-input"
                            value={narrationDraft}
                            onChange={(event) => setNarrationDrafts((current) => ({ ...current, [draftKey]: event.target.value }))}
                            disabled={isSavingAnySegment || isRenderingReview}
                            rows={3}
                          />
                        </label>
                        <button
                          className="ios-button review-segment-save"
                          type="button"
                          disabled={!beat.segment_id || isSavingAnySegment || isRenderingReview || !narrationDirty || narrationDraft.trim().length === 0}
                          onClick={() => persistSegmentEdit(beat, index)}
                          title={beat.segment_id ? "Lưu lời dẫn" : "Job cũ chưa hỗ trợ chỉnh từng segment"}
                        >
                          {isSaving ? <Loader2 className="spin" size={15} /> : <Save size={15} />}
                          Lưu lời dẫn
                        </button>

                        {candidates.length > 0 && (
                          <div className="review-candidates">
                            <span className="review-candidates-label">Cảnh thay thế tốt nhất</span>
                            <div className="review-candidate-grid">
                              {candidates.map((candidate, candidateIndex) => {
                                const candidateScore = normalizedMatchScore(candidate.match_score);
                                return (
                                  <div className="review-candidate" key={candidate.candidate_id || `${candidate.scene_id}-${candidateIndex}`}>
                                    <button
                                      type="button"
                                      className="review-candidate-preview"
                                      disabled={isSavingAnySegment || isRenderingReview}
                                      onClick={() => previewCandidate(index, candidate)}
                                      title="Xem cảnh này trong phim gốc"
                                    >
                                      {candidate.thumbnail_url ? (
                                        <img src={toAbsoluteApiUrl(candidate.thumbnail_url)} alt={`Phương án ${candidateIndex + 1}: ${candidate.scene_id}`} loading="lazy" />
                                      ) : (
                                        <span className="review-thumbnail-empty"><ImageOff size={18} />Xem cảnh</span>
                                      )}
                                    </button>
                                    <div className="review-candidate-meta">
                                      <strong>#{candidateIndex + 1} {candidate.scene_id}</strong>
                                      <span>{formatReviewRange(candidate.start_seconds, candidate.end_seconds)}</span>
                                      {candidateScore != null && <span>{Math.round(candidateScore * 100)}% khớp</span>}
                                      {candidate.match_reason && <p>{candidate.match_reason}</p>}
                                    </div>
                                    <button
                                      type="button"
                                      className="ios-button ios-button-secondary review-candidate-choose"
                                      disabled={!beat.segment_id || isSavingAnySegment || isRenderingReview}
                                      onClick={() => persistSegmentEdit(beat, index, candidate)}
                                    >
                                      {isSaving ? <Loader2 className="spin" size={14} /> : <Check size={14} />}
                                      Dùng cảnh này
                                    </button>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </article>
                    );
                  })}
                </div>
              </div>
              <ReviewBlock title="Text thumbnail" value={result.thumbnail_text} onCopy={copyText} />
              <ReviewBlock title="Hashtag" value={result.tags.map((tag) => `#${tag.replace(/^#+/, "")}`).join(" ")} onCopy={copyText} />
            </div>
          )}
          </section>
        </div>
      </div>
    </div>
  );
});

function ReviewDisplayControls({
  options,
  watermarkName,
  isLogoUploading,
  isDetectingAI,
  onFieldChange,
  onOpenLogoPicker,
  onDetectAI,
  onAddBlurBox,
}: {
  options: ReviewRenderOptions;
  watermarkName: string;
  isLogoUploading: boolean;
  isDetectingAI: boolean;
  onFieldChange: <K extends keyof ReviewRenderOptions>(key: K, value: ReviewRenderOptions[K]) => void;
  onOpenLogoPicker: () => void;
  onDetectAI: () => void;
  onAddBlurBox: () => void;
}) {
  return (
    <div className="review-block review-display-controls">
      <div className="review-block-head">
        <h3>Chỉnh hiển thị video review</h3>
        <button className="ios-button ios-button-secondary" type="button" onClick={onDetectAI} disabled={isDetectingAI}>
          {isDetectingAI ? <Loader2 className="spin" size={15} /> : <Wand2 size={15} />}
          AI nhận diện
        </button>
      </div>

      <div className="review-control-checks">
        <ReviewCheckBox
          checked={options.hard_subtitles}
          label="Ghi phụ đề cứng"
          onChange={() => onFieldChange("hard_subtitles", !options.hard_subtitles)}
        />
        <ReviewCheckBox
          checked={options.subtitle_box_enabled}
          label="Nền mờ sau phụ đề"
          onChange={() => onFieldChange("subtitle_box_enabled", !options.subtitle_box_enabled)}
        />
        <ReviewCheckBox
          checked={options.blur_box_enabled}
          label="Ẩn phụ đề gốc để chỉ còn 1 dòng review"
          onChange={() => onFieldChange("blur_box_enabled", !options.blur_box_enabled)}
        />
        <ReviewCheckBox
          checked={options.logo_enabled}
          label="Chèn logo"
          onChange={() => onFieldChange("logo_enabled", !options.logo_enabled)}
        />
        <ReviewCheckBox
          checked={options.cinematic_bars_enabled}
          label="Dải đen cinematic"
          onChange={() => onFieldChange("cinematic_bars_enabled", !options.cinematic_bars_enabled)}
        />
      </div>

      <div className="editor-grid review-editor-grid">
        <ReviewRangeField label="Sub ngang" value={options.subtitle_x_percent} min={0} max={100} suffix="%" onChange={(value) => onFieldChange("subtitle_x_percent", value)} />
        <ReviewRangeField label="Sub dọc" value={options.subtitle_y_percent} min={8} max={94} suffix="%" onChange={(value) => onFieldChange("subtitle_y_percent", value)} />
        <ReviewRangeField label="Cỡ sub" value={options.subtitle_font_size} min={16} max={120} suffix="px" onChange={(value) => onFieldChange("subtitle_font_size", value)} />
        <ReviewRangeField label="Cao nền sub" value={options.subtitle_box_height_percent} min={8} max={45} suffix="%" onChange={(value) => onFieldChange("subtitle_box_height_percent", value)} />
        <ReviewRangeField label="Độ mờ nền" value={options.subtitle_box_opacity} min={0} max={100} suffix="%" onChange={(value) => onFieldChange("subtitle_box_opacity", value)} />
      </div>

      {options.blur_box_enabled && (
        <div className="editor-grid review-editor-grid">
          <ReviewRangeField label="Vị trí mờ chữ" value={options.blur_box_y_percent} min={0} max={95} suffix="%" onChange={(value) => onFieldChange("blur_box_y_percent", value)} />
          <ReviewRangeField label="Cao vùng mờ" value={options.blur_box_height_percent} min={5} max={45} suffix="%" onChange={(value) => onFieldChange("blur_box_height_percent", value)} />
        </div>
      )}

      <div className="review-logo-tools">
        <button type="button" className="ios-button" onClick={onOpenLogoPicker} disabled={isLogoUploading}>
          {isLogoUploading ? <Loader2 className="spin" size={16} /> : <FileImage size={16} />}
          Chọn logo
        </button>
        <span>{watermarkName}</span>
      </div>

      {options.logo_enabled && (
        <div className="editor-grid review-editor-grid">
          <ReviewRangeField label="Logo ngang" value={options.logo_x_percent} min={0} max={100} suffix="%" onChange={(value) => onFieldChange("logo_x_percent", value)} />
          <ReviewRangeField label="Logo dọc" value={options.logo_y_percent} min={0} max={100} suffix="%" onChange={(value) => onFieldChange("logo_y_percent", value)} />
          <ReviewRangeField label="Rộng logo" value={options.logo_width} min={32} max={260} suffix="px" onChange={(value) => onFieldChange("logo_width", value)} />
        </div>
      )}

      {options.cinematic_bars_enabled && (
        <ReviewRangeField label="Cao dải đen" value={options.cinematic_bars_height_percent} min={0} max={30} suffix="%" onChange={(value) => onFieldChange("cinematic_bars_height_percent", value)} />
      )}

      <button type="button" className="ios-button ios-button-secondary review-add-blur" onClick={onAddBlurBox}>
        <Plus size={15} />
        Thêm vùng mờ kéo tay
      </button>
      <p className="hint-text blur-hint">Mẹo: có thể kéo trực tiếp phụ đề, logo và vùng mờ trên Live View; các chỉnh này áp dụng cho lần tạo video review tiếp theo.</p>
    </div>
  );
}

function ReviewRangeField({
  label,
  value,
  min,
  max,
  suffix,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  suffix: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="range-field">
      <span>
        {label}
        <strong>{value}{suffix}</strong>
      </span>
      <input type="range" min={min} max={max} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

function ReviewCheckBox({ checked, label, onChange }: { checked: boolean; label: string; onChange: () => void }) {
  return (
    <label className="check-item">
      <input type="checkbox" checked={checked} onChange={onChange} />
      <span>{checked && <Check size={14} />}</span>
      {label}
    </label>
  );
}

const ReviewLivePreview = React.forwardRef<HTMLVideoElement, {
  mode: "source" | "output";
  onModeChange: (mode: "source" | "output") => void;
  sourceVideoUrl: string | null;
  outputVideoUrl: string | null;
  activeVideoUrl: string | null;
  selectedBeat: ReviewBeat | null;
  outputDurationSeconds: number | null;
  renderOptions: ReviewRenderOptions;
  onRenderFieldChange: <K extends keyof ReviewRenderOptions>(key: K, value: ReviewRenderOptions[K]) => void;
  onOutputDurationChange: (duration: number) => void;
}>(function ReviewLivePreview({ mode, onModeChange, sourceVideoUrl, outputVideoUrl, activeVideoUrl, selectedBeat, outputDurationSeconds, renderOptions, onRenderFieldChange, onOutputDurationChange }, ref) {
  return (
    <section className="ios-card review-live-card">
      <div className="section-title">
        <span>Live</span>
        <h2>Live view phim</h2>
      </div>
      <div className="review-preview-tabs">
        <button type="button" className={mode === "source" ? "active" : ""} onClick={() => onModeChange("source")} disabled={!sourceVideoUrl}>
          Phim gốc
        </button>
        <button type="button" className={mode === "output" ? "active" : ""} onClick={() => onModeChange("output")} disabled={!outputVideoUrl}>
          Video review
        </button>
      </div>
      {mode === "source" ? (
        sourceVideoUrl ? (
          <LivePreview
            {...renderOptions}
            watermark_file_name={renderOptions.watermark_file_name ?? null}
            previewVideoUrl={sourceVideoUrl}
            setField={onRenderFieldChange}
            videoRef={ref}
          />
        ) : (
          <div className="video-frame review-live-frame">
            <div className="review-live-empty">
              <Video size={36} />
              <strong>Import phim để xem live view</strong>
              <span>Khi có phim, bạn có thể kéo logo, sub và vùng làm mờ trực tiếp.</span>
            </div>
          </div>
        )
      ) : (
        <div className="video-frame review-live-frame">
          {activeVideoUrl ? (
          <video
            key={activeVideoUrl}
            ref={ref}
            className="preview-video-element"
            src={activeVideoUrl}
            controls
            playsInline
            onLoadedMetadata={(event) => {
              if (mode === "output") onOutputDurationChange(event.currentTarget.duration);
            }}
          />
          ) : (
            <div className="review-live-empty">
              <Video size={36} />
              <strong>Chưa có video review</strong>
              <span>Tạo bản review xong tab này sẽ xem được output.</span>
            </div>
          )}
        </div>
      )}
      <div className="review-preview-meta">
        <strong>{mode === "output" ? "Đang xem video review đã render" : "Đang xem phim gốc để so cảnh"}</strong>
        {mode === "output" && outputDurationSeconds != null && (
          <span>Thời lượng video hoàn thành: <strong>{formatReviewDuration(outputDurationSeconds)}</strong></span>
        )}
        {selectedBeat ? (
          <span>{selectedBeat.time_hint} • {selectedBeat.purpose}</span>
        ) : (
          <span>Click một beat bên dưới để nhảy tới đoạn cần kiểm tra.</span>
        )}
      </div>
    </section>
  );
});

function ReviewBlock({ title, value, onCopy, large = false }: { title: string; value: string; onCopy: (value: string) => void; large?: boolean }) {
  return (
    <div className="review-block">
      <div className="review-block-head">
        <h3>{title}</h3>
        <button type="button" className="ios-button ios-button-secondary" onClick={() => onCopy(value)}>
          <Copy size={15} />
          Copy
        </button>
      </div>
      <p className={large ? "review-script" : undefined}>{value}</p>
    </div>
  );
}

function ReviewQaMetric({ label, value }: { label: string; value: number }) {
  const safeValue = formatQaScore(value);
  return (
    <div className="review-qa-metric">
      <span>{label}<strong>{safeValue}%</strong></span>
      <div><i style={{ width: `${safeValue}%` }} /></div>
    </div>
  );
}

function formatQaScore(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

function reviewBeatKey(beat: ReviewBeat, index: number): string {
  return beat.segment_id || `legacy-${index}`;
}

function normalizedMatchScore(value: number | null | undefined): number | null {
  if (value == null || !Number.isFinite(value)) return null;
  const normalized = value > 1 ? value / 100 : value;
  return Math.max(0, Math.min(1, normalized));
}

function formatReviewRange(
  start: number | null | undefined,
  end: number | null | undefined,
  fallback = "Chưa có mốc",
): string {
  if (start == null || !Number.isFinite(start)) return fallback;
  if (end == null || !Number.isFinite(end)) return formatReviewTimestamp(start);
  return `${formatReviewTimestamp(start)} – ${formatReviewTimestamp(end)}`;
}

function formatReviewTimestamp(value: number): string {
  const safeValue = Math.max(0, Math.floor(value));
  const hours = Math.floor(safeValue / 3600);
  const minutes = Math.floor((safeValue % 3600) / 60);
  const seconds = safeValue % 60;
  if (hours > 0) return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function formatReviewDuration(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0 giây";
  const totalSeconds = Math.max(1, Math.round(value));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  const parts: string[] = [];
  if (hours > 0) parts.push(`${hours} giờ`);
  if (minutes > 0) parts.push(`${minutes} phút`);
  if (seconds > 0 || parts.length === 0) parts.push(`${seconds} giây`);
  return parts.join(" ");
}

function normalizeTargetMinutes(value: number): number {
  const rounded = Number.isFinite(value) ? Math.round(value) : 8;
  return Math.max(MIN_REVIEW_MINUTES, Math.min(MAX_REVIEW_MINUTES, rounded));
}

function reviewStyleName(value: ReviewDraftRequest["style"]): string {
  return reviewStyles.find((item) => item.value === value)?.label ?? value;
}

function processingModeLabel(value: ProcessingMode | null | undefined): string {
  if (value === "fast") return "Siêu nhanh";
  if (value === "quality") return "Đẹp nhất";
  return "Cân bằng";
}

function firstFinitePositive(values: Array<number | null | undefined>): number | null {
  const value = values.find((item) => item != null && Number.isFinite(item) && item > 0);
  return value ?? null;
}

function reviewPlannedDurationSeconds(result: ReviewDraftJob["result"]): number | null {
  if (!result) return null;
  const reported = firstFinitePositive([
    result.narration_duration_seconds,
  ]);
  if (reported != null) return reported;

  const voiceEnds = result.beats
    .map((beat) => beat.voice_end ?? beat.voice_end_seconds)
    .filter((value): value is number => value != null && Number.isFinite(value) && value > 0);
  return voiceEnds.length > 0 ? Math.max(...voiceEnds) : null;
}

function assessReviewDuration(targetSeconds: number | null, actualSeconds: number | null): {
  significant: boolean;
  direction: "short" | "long";
  absoluteDeltaSeconds: number;
  percentOfTarget: number;
} | null {
  if (targetSeconds == null || actualSeconds == null || targetSeconds <= 0 || actualSeconds <= 0) return null;
  const delta = actualSeconds - targetSeconds;
  const absoluteDeltaSeconds = Math.abs(delta);
  return {
    significant: absoluteDeltaSeconds / targetSeconds > 0.10,
    direction: delta < 0 ? "short" : "long",
    absoluteDeltaSeconds,
    percentOfTarget: Math.max(1, Math.round(actualSeconds / targetSeconds * 100)),
  };
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exponent;
  const digits = exponent === 0 ? 0 : value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[exponent]}`;
}

function fileNameFromPath(value: string): string {
  const normalized = value.replace(/\\/g, "/");
  return normalized.split("/").pop() || value || "Phim đã import";
}

function sourceUrlFromPath(value: string): string | null {
  const normalized = value.replace(/\\/g, "/");
  const marker = "/uploads/videos/";
  const markerIndex = normalized.lastIndexOf(marker);
  if (markerIndex < 0) return null;
  const fileName = normalized.slice(markerIndex + marker.length);
  return fileName ? toAbsoluteApiUrl(`/api/uploads/video/${encodeURIComponent(fileName)}`) : null;
}

function renderOptionsFromRequest(request: Partial<ReviewDraftRequest>): ReviewRenderOptions {
  const boxes: CustomBlurBox[] = Array.isArray(request.custom_blur_boxes)
    ? request.custom_blur_boxes.map((box) => ({ ...box }))
    : [];

  return {
    ...defaultReviewRenderOptions,
    hard_subtitles: request.hard_subtitles ?? defaultReviewRenderOptions.hard_subtitles,
    subtitle_x_percent: request.subtitle_x_percent ?? defaultReviewRenderOptions.subtitle_x_percent,
    subtitle_y_percent: request.subtitle_y_percent ?? defaultReviewRenderOptions.subtitle_y_percent,
    subtitle_font_size: request.subtitle_font_size ?? defaultReviewRenderOptions.subtitle_font_size,
    subtitle_box_enabled: request.subtitle_box_enabled ?? defaultReviewRenderOptions.subtitle_box_enabled,
    subtitle_box_opacity: request.subtitle_box_opacity ?? defaultReviewRenderOptions.subtitle_box_opacity,
    subtitle_box_height_percent: request.subtitle_box_height_percent ?? defaultReviewRenderOptions.subtitle_box_height_percent,
    logo_enabled: request.logo_enabled ?? defaultReviewRenderOptions.logo_enabled,
    logo_width: request.logo_width ?? defaultReviewRenderOptions.logo_width,
    logo_x_percent: request.logo_x_percent ?? defaultReviewRenderOptions.logo_x_percent,
    logo_y_percent: request.logo_y_percent ?? defaultReviewRenderOptions.logo_y_percent,
    cinematic_bars_enabled: request.cinematic_bars_enabled ?? defaultReviewRenderOptions.cinematic_bars_enabled,
    cinematic_bars_height_percent: request.cinematic_bars_height_percent ?? defaultReviewRenderOptions.cinematic_bars_height_percent,
    blur_box_enabled: request.blur_box_enabled ?? defaultReviewRenderOptions.blur_box_enabled,
    blur_box_y_percent: request.blur_box_y_percent ?? defaultReviewRenderOptions.blur_box_y_percent,
    blur_box_height_percent: request.blur_box_height_percent ?? defaultReviewRenderOptions.blur_box_height_percent,
    custom_blur_boxes: boxes,
    watermark_file_name: request.watermark_file_name ?? null,
    output_resolution: request.output_resolution ?? defaultReviewRenderOptions.output_resolution,
  };
}

function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString("vi-VN", { hour: "2-digit", minute: "2-digit", day: "2-digit", month: "2-digit" });
}
