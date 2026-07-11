import React, { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState, useCallback } from "react";
import {
  BadgeCheck,
  Check,
  ChevronRight,
  Copy,
  Download,
  FileImage,
  Loader2,
  Play,
  Trash2,
  Upload,
  Video,
  Wand2,
  Youtube,
} from "lucide-react";
import { cancelJob, clearJobs, createJob, detectBlurRegions, fetchUrlPreview, generateJobMetadata, getJob, listJobs, toAbsoluteApiUrl, uploadVideo, uploadWatermark } from "./lib/api";
import type { UploadProgress } from "./lib/api";
import type { BgmMode, DubbingRequest, JobProgress, VoiceGender } from "./types/api";
import { YoutubeStats } from "./components/YoutubeStats";
import { LivePreview } from "./components/LivePreview";
import { MovieReview } from "./components/MovieReview";

const languages = [
  { value: "auto", label: "Tự nhận diện" },
  { value: "en", label: "Tiếng Anh" },
  { value: "zh", label: "Tiếng Trung" },
  { value: "vi", label: "Tiếng Việt" },
] as const;

const defaultForm: DubbingRequest = {
  source_url: "",
  local_file_path: "",
  voice_gender: "female",
  bgm_mode: "demucs",
  use_demucs: true,
  video_speed: 1.0,
  auto_publish: [],
  clone_voice: false,
  hard_subtitles: true,
  source_has_hard_subtitles: false,
  subtitle_x_percent: 50,
  subtitle_y_percent: 78,
  subtitle_font_size: 48,
  subtitle_box_enabled: true,
  subtitle_box_opacity: 55,
  subtitle_box_height_percent: 20,
  source_language: "auto",
  ducking_volume_db: -12,
  output_resolution: "original",
  logo_enabled: true,
  logo_width: 96,
  logo_x_percent: 94,
  logo_y_percent: 6,
  cinematic_bars_enabled: false,
  cinematic_bars_height_percent: 10,
  blur_box_enabled: false,
  blur_box_y_percent: 80,
  blur_box_height_percent: 15,
  custom_blur_boxes: [],
  watermark_file_name: null,
};

function normalizeBlurBand(yPercent: number, heightPercent: number) {
  const height = Math.max(8, Math.min(18, Math.round(heightPercent)));
  const y = Math.max(74, Math.min(90, Math.round(yPercent)));
  const safeY = Math.min(y, 97 - height);
  return {
    y: safeY,
    height,
    subtitleY: Math.max(82, Math.min(90, Math.round(safeY + height / 2 - 1))),
  };
}

function suggestSubtitleFontSize(blurHeightPercent: number) {
  return Math.max(32, Math.min(42, Math.round(blurHeightPercent * 3.8 + 4)));
}

function needsYoutubeMetadata(job: JobProgress): boolean {
  if (job.status !== "completed") return false;
  const title = job.seo_title?.trim();
  const tags = job.seo_tags ?? [];
  return !title || title === "Video đã được xử lý" || tags.length === 0;
}

interface DebouncedInputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "onChange"> {
  value: string;
  onChange: (val: string) => void;
  debounceMs?: number;
}

function DebouncedInput({ value, onChange, debounceMs = 300, ...props }: DebouncedInputProps) {
  const [localValue, setLocalValue] = useState(value);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setLocalValue(val);
    if (timerRef.current) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => {
      onChange(val);
    }, debounceMs);
  };

  return <input {...props} value={localValue} onChange={handleChange} />;
}

export function App() {
  const [form, setForm] = useState<DubbingRequest>(defaultForm);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<JobProgress | null>(null);
  const [jobs, setJobs] = useState<JobProgress[]>([]);
  const [isSubmitting, setSubmitting] = useState(false);
  const [isUploading, setUploading] = useState(false);
  const [isVideoUploading, setVideoUploading] = useState(false);
  const [videoUploadProgress, setVideoUploadProgress] = useState<UploadProgress | null>(null);
  const [isClearingJobs, setClearingJobs] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [watermarkName, setWatermarkName] = useState("Chưa có logo được tải lên");
  const [videoName, setVideoName] = useState("Chưa có video được import");
  const [previewVideoUrl, setPreviewVideoUrl] = useState<string | null>(null);
  const [isPreviewLoading, setPreviewLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"workspace" | "youtube" | "review">("workspace");

  const videoInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    return () => {
      if (previewVideoUrl && previewVideoUrl.startsWith("blob:")) {
        URL.revokeObjectURL(previewVideoUrl);
      }
    };
  }, [previewVideoUrl]);

  const canSubmit = useMemo(() => {
    return Boolean(form.source_url?.trim() || form.local_file_path?.trim()) && !isSubmitting;
  }, [form.local_file_path, form.source_url, isSubmitting]);

  const refreshJobs = useCallback(async () => {
    const nextJobs = await listJobs();
    setJobs(nextJobs);
    return nextJobs;
  }, []);

  useEffect(() => {
    refreshJobs().catch(() => undefined);
  }, [refreshJobs]);

  useEffect(() => {
    if (!activeJobId) return;

    let cancelled = false;
    const poll = async () => {
      try {
        let job = await getJob(activeJobId);
        if (needsYoutubeMetadata(job)) {
          job = await generateJobMetadata(job.job_id);
        }
        if (cancelled) return;
        setActiveJob(job);
        setJobs((previous) => [job, ...previous.filter((item) => item.job_id !== job.job_id)].slice(0, 8));
      } catch (error) {
        if (!cancelled) setMessage(error instanceof Error ? error.message : "Không thể lấy trạng thái job.");
      }
    };

    poll();
    const timer = window.setInterval(poll, 1300);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeJobId]);

  const setField = useCallback(<K extends keyof DubbingRequest>(key: K, value: DubbingRequest[K]) => {
    setForm((current) => ({ ...current, [key]: value }));
  }, []);

  async function handleWatermark(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setMessage(null);
    try {
      const uploaded = await uploadWatermark(file);
      setField("watermark_file_name", uploaded.file_name);
      setWatermarkName(file.name);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Tải logo thất bại.");
    } finally {
      setUploading(false);
    }
  }

  async function handleVideoImport(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    setVideoUploading(true);
    setVideoUploadProgress({
      loaded: 0,
      total: file.size || null,
      percent: 0,
      bytesPerSecond: null,
    });
    setMessage(null);
    try {
      const uploaded = await uploadVideo(file, (progress) => {
        setVideoUploadProgress(progress);
      });
      if (!uploaded.local_file_path) {
        throw new Error("Backend không trả về đường dẫn video đã import.");
      }
      setForm((current) => ({
        ...current,
        source_url: "",
        local_file_path: uploaded.local_file_path,
      }));
      setVideoName(file.name);
      setPreviewVideoUrl(URL.createObjectURL(file));
    } catch (error) {
      setVideoUploadProgress(null);
      setMessage(error instanceof Error ? error.message : "Import video thất bại.");
    } finally {
      setVideoUploading(false);
      event.target.value = "";
    }
  }

  const openVideoPicker = useCallback(() => {
    if (isVideoUploading) return;
    videoInputRef.current?.click();
  }, [isVideoUploading]);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setMessage(null);

    const payload: DubbingRequest = {
      ...form,
      source_url: form.source_url?.trim() || null,
      local_file_path: form.local_file_path?.trim() || null,
    };
    if (payload.blur_box_enabled) {
      const normalized = normalizeBlurBand(payload.blur_box_y_percent, payload.blur_box_height_percent);
      payload.blur_box_y_percent = normalized.y;
      payload.blur_box_height_percent = normalized.height;
    }

    try {
      const response = await createJob(payload);
      setActiveJobId(response.job_id);
      setMessage("Đã gửi job. Backend sẽ render đúng vị trí phụ đề trong live view.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Không thể tạo job.");
    } finally {
      setSubmitting(false);
    }
  }

  const handleClearJobs = useCallback(async () => {
    const confirmed = window.confirm("Đưa toàn bộ file job và video đã render vào Thùng rác, đồng thời xoá lịch sử job?");
    if (!confirmed) return;

    setClearingJobs(true);
    setMessage(null);
    try {
      const result = await clearJobs();
      setJobs([]);
      setActiveJob(null);
      setActiveJobId(null);
      await refreshJobs();
      setMessage(result.message || "Đã xoá lịch sử job và đưa file vào Thùng rác.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Không thể xoá lịch sử job.");
    } finally {
      setClearingJobs(false);
    }
  }, [refreshJobs]);

  const handleSelectJob = useCallback((id: string) => {
    setActiveJobId(id);
  }, []);

  const handleAddBlurBox = useCallback(() => {
    const boxes = [...form.custom_blur_boxes];
    boxes.push({ x_percent: 40, y_percent: 40, width_percent: 20, height_percent: 20 });
    setField("custom_blur_boxes", boxes);
  }, [form.custom_blur_boxes, setField]);

  const [isDetectingAI, setIsDetectingAI] = useState(false);

  const handleAIDetect = useCallback(async () => {
    const videoPath = form.local_file_path?.trim();
    if (!videoPath) {
      setMessage("Vui lòng import video trước khi phân tích AI.");
      return;
    }
    setIsDetectingAI(true);
    setMessage("AI đang phân tích khung hình và kiểm tra lại cấu hình...");
    try {
      const result = await detectBlurRegions(videoPath, true, true, "ai");
      if (result.count === 0 && !result.auto_logo && !result.review?.ok) {
        setMessage("AI không phát hiện ra vùng cần làm mờ nào.");
      } else {
        // Preset AI tong the: render ra phai dung nhu preview, che sach chu goc va giu phu de moi ro.
        setField("hard_subtitles", true);
        setField("source_has_hard_subtitles", false);
        setField("source_language", "auto");
        setField("voice_gender", "female");
        setField("bgm_mode", "demucs");
        setField("use_demucs", true);
        setField("video_speed", 1);
        setField("output_resolution", "original");
        setField("cinematic_bars_enabled", false);
        setField("subtitle_x_percent", 50);
        setField("subtitle_font_size", 40);
        setField("subtitle_box_enabled", false);
        setField("subtitle_box_opacity", 0);
        setField("subtitle_box_height_percent", 22);
        setField("auto_publish", []);

        // 1. Cập nhật các vùng làm mờ mới
        if (result.config) {
          const normalized = normalizeBlurBand(
            result.config.blur_box_y_percent,
            result.config.blur_box_height_percent,
          );
          const subtitleFontSize = suggestSubtitleFontSize(normalized.height);
          setField("blur_box_enabled", true);
          setField("blur_box_y_percent", normalized.y);
          setField("blur_box_height_percent", normalized.height);
          setField("custom_blur_boxes", result.config.custom_blur_boxes);
          setField("subtitle_font_size", subtitleFontSize);

          if (typeof result.config.subtitle_y_percent === "number") {
            setField("subtitle_y_percent", Math.max(82, Math.min(90, Math.round(result.config.subtitle_y_percent - 1))));
          } else {
            setField("subtitle_y_percent", normalized.subtitleY);
          }
        } else {
          const normalized = normalizeBlurBand(82, 12);
          const subtitleFontSize = suggestSubtitleFontSize(normalized.height);
          setField("blur_box_enabled", true);
          setField("blur_box_y_percent", normalized.y);
          setField("blur_box_height_percent", normalized.height);
          setField("subtitle_y_percent", normalized.subtitleY);
          setField("subtitle_font_size", subtitleFontSize);
          setField("custom_blur_boxes", result.regions.map(r => ({
            x_percent: r.x_percent,
            y_percent: r.y_percent,
            width_percent: r.width_percent,
            height_percent: r.height_percent,
          })));
        }

        // 2. Tự động dịch chuyển phụ đề mới đè lên vùng mờ chữ Trung dưới đáy
        const subRegion = result.config ? undefined : result.regions.find(r => r.label?.toLowerCase().includes("sub"));
        if (subRegion) {
          // Tính tâm dọc của vùng mờ để đặt chữ phụ đề đè lên
          const targetY = Math.max(82, Math.min(90, Math.round(subRegion.y_percent + subRegion.height_percent / 2 - 1)));
          setField("subtitle_y_percent", targetY);
        }

        // 3. Tự động chèn logo kênh sang góc trái (nếu tìm thấy logo trong thư mục Pictures)
        if (result.auto_logo) {
          setField("watermark_file_name", result.auto_logo.watermark_file_name);
          setField("logo_width", result.auto_logo.logo_width);
          setField("logo_x_percent", result.auto_logo.logo_x_percent);
          setField("logo_y_percent", result.auto_logo.logo_y_percent);
          setField("logo_enabled", result.auto_logo.logo_enabled);
          setWatermarkName("logo kenh.png");
        }

        const notes = result.review?.notes?.length ? ` ${result.review.notes.join(" ")}` : "";
        setMessage(`AI đã tự cấu hình đầy đủ: phụ đề, giọng đọc, Demucs, thanh mờ chữ gốc, logo và metadata YouTube. ${result.count} vùng hợp lý.${notes}`);
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "AI phân tích thất bại.");
    } finally {
      setIsDetectingAI(false);
    }
  }, [form.local_file_path, setField, setWatermarkName]);

  return (
    <main className="ios-container">
      <nav className="ios-navbar ios-glass">
        <a className="brand" href="/" aria-label="Auto-Translate AI">
          <span className="brand-mark">
            <Wand2 size={21} />
          </span>
          <span>Auto-Translate AI</span>
        </a>
        <div className="nav-links">
          <span className="health-pill">
            <BadgeCheck size={16} />
            Backend đang chạy
          </span>
          <button
            className={`nav-tab ios-button ios-button-secondary ${activeTab === 'youtube' ? 'active' : ''}`}
            onClick={() => setActiveTab('youtube')}
          >
            <Youtube size={16} /> YouTube Stats
          </button>
          <button
            className={`nav-tab ios-button ios-button-secondary ${activeTab === 'review' ? 'active' : ''}`}
            onClick={() => setActiveTab('review')}
          >
            Review phim
          </button>
          <button
            className={`nav-tab ios-button ios-button-secondary ${activeTab === 'workspace' ? 'active' : ''}`}
            onClick={() => setActiveTab('workspace')}
          >
            Workspace
          </button>
          <a className="nav-tab nav-doc-link ios-button ios-button-secondary" href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">
            Tài liệu API
            <ChevronRight size={16} />
          </a>
        </div>
      </nav>

      {activeTab === 'youtube' ? (
        <div key="youtube" className="ios-view-transition">
          <YoutubeStats />
        </div>
      ) : activeTab === 'review' ? (
        <MovieReview />
      ) : (
        <div key="workspace" className="ios-view-transition">
          <header className="page-heading">
            <div>
              <h1>Video dubbing workspace</h1>
              <p>Cấu hình, kéo vị trí phụ đề và render video trực quan</p>
            </div>
            <button className="ios-button" type="submit" form="dubbing-form" disabled={!canSubmit}>
              {isSubmitting ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
              Render video
            </button>
          </header>

          <form id="dubbing-form" className="workspace" onSubmit={handleSubmit}>
            <div className="controls-column">
              <section className="ios-card">
                <SectionTitle index="01" title="Nguồn video" />
                <label className="field">
                  <span>URL video</span>
                  <div className="url-preview-row">
                    <DebouncedInput
                      className="ios-input"
                      value={form.source_url ?? ""}
                      onChange={(val) => {
                        setField("source_url", val);
                        if (val.trim() && val.trim().toLowerCase().endsWith(".mp4")) {
                          setPreviewVideoUrl(val);
                        } else if (val.trim() === "") {
                          setPreviewVideoUrl(null);
                        }
                      }}
                      placeholder="YouTube, TikTok hoặc Douyin"
                    />
                    {form.source_url && !form.source_url.toLowerCase().endsWith(".mp4") && (
                      <button
                        type="button"
                        className="preview-load-button ios-button ios-button-secondary"
                        disabled={isPreviewLoading}
                        onClick={async () => {
                          setPreviewLoading(true);
                          setMessage(null);
                          try {
                            const res = await fetchUrlPreview(form.source_url!);
                            setPreviewVideoUrl(toAbsoluteApiUrl(res.url));
                          } catch (e: any) {
                            setMessage(e.message);
                          } finally {
                            setPreviewLoading(false);
                          }
                        }}
                      >
                        {isPreviewLoading ? <Loader2 size={18} className="spin" /> : "Tải bản xem trước"}
                      </button>
                    )}
                  </div>
                </label>
                <div className="field">
                  <span>File nội bộ</span>
                  <div className="upload-row">
                    <button className="ios-button" type="button" onClick={openVideoPicker} disabled={isVideoUploading}>
                      {isVideoUploading ? <Loader2 className="spin" size={17} /> : <Upload size={17} />}
                      <span>{isVideoUploading ? "Dang import..." : "Import video"}</span>
                    </button>
                    <input
                      ref={videoInputRef}
                      className="file-input-hidden"
                      type="file"
                      accept="video/*,.mp4,.mov,.mkv,.m4v,.webm"
                      onChange={handleVideoImport}
                    />
                    <span className="upload-name">
                      <Video size={17} />
                      {videoName}
                    </span>
                  </div>
                  {videoUploadProgress && (
                    <div className="upload-progress">
                      <div className="upload-progress-meta">
                        <strong>
                          {isVideoUploading
                            ? videoUploadProgress.percent === 100
                              ? "Da gui xong, dang cho backend xac nhan..."
                              : `Dang import ${videoUploadProgress.percent ?? 0}%`
                            : "Import hoan tat"}
                        </strong>
                        <span>
                          {formatBytes(videoUploadProgress.loaded)}
                          {videoUploadProgress.total ? ` / ${formatBytes(videoUploadProgress.total)}` : ""}
                          {videoUploadProgress.bytesPerSecond ? ` • ${formatUploadSpeed(videoUploadProgress.bytesPerSecond)}` : ""}
                        </span>
                      </div>
                      <div className="upload-progress-track">
                        <div
                          className="upload-progress-bar"
                          style={{ width: `${videoUploadProgress.percent ?? 0}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>

                <div className="two-fields">
                  <label className="field">
                    <span>Ngôn ngữ gốc</span>
                    <select className="ios-input" value={form.source_language} onChange={(event) => setField("source_language", event.target.value as DubbingRequest["source_language"])}>
                      {languages.map((language) => (
                        <option key={language.value} value={language.value}>
                          {language.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    <span>Ducking (dB)</span>
                    <DebouncedInput
                      className="ios-input"
                      type="number"
                      min={-36}
                      max={0}
                      value={String(form.ducking_volume_db)}
                      onChange={(val) => setField("ducking_volume_db", Number(val))}
                    />
                  </label>
                </div>
              </section>

              <section className="ios-card">
                <SectionTitle index="02" title="Âm thanh & Giọng đọc" />
                <SegmentedControl<VoiceGender>
                  label="Giọng đọc"
                  value={form.voice_gender}
                  options={[
                    { label: "Nữ", value: "female" },
                    { label: "Nam", value: "male" },
                  ]}
                  onChange={(value) => setField("voice_gender", value)}
                />
                <SegmentedControl<BgmMode>
                  label="Nhạc nền"
                  value={form.bgm_mode}
                  options={[
                    { label: "Giữ âm gốc", value: "demucs" },
                    { label: "Nền nhẹ", value: "ducking" },
                    { label: "Tắt nền", value: "none" },
                  ]}
                  onChange={(value) => setField("bgm_mode", value)}
                />
                {form.bgm_mode !== "none" && (
                  <div className="demucs-toggle">
                    <CheckBox 
                      checked={form.use_demucs} 
                      label="Dùng AI Demucs để tách sạch âm thanh (Render chậm hơn)" 
                      onChange={() => setField("use_demucs", !form.use_demucs)} 
                    />
                  </div>
                )}
              </section>

              <section className="ios-card">
                <SectionTitle index="03" title="Tuỳ chọn Video & Hình ảnh" />
                <div className="two-fields video-options-grid">
                  <label className="field">
                    <span>Độ phân giải xuất</span>
                    <select className="ios-input" value={form.output_resolution} onChange={(event) => setField("output_resolution", event.target.value)}>
                      <option value="original">Giữ nguyên bản gốc</option>
                      <option value="720p">HD (720p)</option>
                      <option value="1080p">Full HD (1080p)</option>
                      <option value="1440p">2K (1440p)</option>
                      <option value="4k">4K (2160p)</option>
                    </select>
                  </label>
                  <label className="field">
                    <span>Tốc độ Video</span>
                    <select className="ios-input" value={form.video_speed} onChange={(event) => setField("video_speed", Number(event.target.value))}>
                      <option value={0.75}>Chậm (0.75x)</option>
                      <option value={1.0}>Bình thường (1x)</option>
                      <option value={1.25}>Nhanh (1.25x)</option>
                      <option value={1.5}>Rất nhanh (1.5x)</option>
                      <option value={2.0}>Siêu nhanh (2x)</option>
                    </select>
                  </label>
                </div>

                <div className="check-grid">
                  <CheckBox checked={form.hard_subtitles} label="Ghi phụ đề cứng vào video" onChange={() => setField("hard_subtitles", !form.hard_subtitles)} />
                  <CheckBox
                    checked={form.source_has_hard_subtitles}
                    label="Video nguồn đã có phụ đề sẵn"
                    onChange={() => setField("source_has_hard_subtitles", !form.source_has_hard_subtitles)}
                  />
                  <CheckBox checked={form.cinematic_bars_enabled} label="Dải đen viền video (Cinematic bars)" onChange={() => setField("cinematic_bars_enabled", !form.cinematic_bars_enabled)} />
                  <CheckBox checked={form.blur_box_enabled} label="Thanh làm mờ chữ gốc" onChange={() => setField("blur_box_enabled", !form.blur_box_enabled)} />
                  <CheckBox 
                    checked={form.logo_enabled} 
                    label="Đóng dấu Logo Watermark" 
                    onChange={() => setField("logo_enabled", !form.logo_enabled)} 
                  />
                  <CheckBox 
                    checked={form.auto_publish?.includes('youtube') ?? false} 
                    label="Tự động đăng YouTube sau khi render" 
                    onChange={() => {
                      const isEnabled = form.auto_publish?.includes('youtube');
                      setField('auto_publish', isEnabled ? (form.auto_publish || []).filter(t => t !== 'youtube') : [...(form.auto_publish || []), 'youtube' as any]);
                    }} 
                  />
                </div>

                <div className="ai-tools">
                  {/* Phân tích tự động bằng trí tuệ nhân tạo (AI) */}
                  <div className="ai-assist-card">
                    <div className="ai-assist-title">
                      <span>🤖</span> Tự động bằng Trí tuệ nhân tạo (AI)
                    </div>
                    <p>
                      AI sẽ tự phân tích khung hình video để xác định các vùng cần làm mờ và tự động cấu hình phù hợp.
                    </p>
                    <button
                      type="button"
                      className="ios-button"
                      onClick={handleAIDetect}
                      disabled={isDetectingAI || !form.local_file_path}
                    >
                      {isDetectingAI ? <Loader2 className="spin" size={16} /> : <span>🤖</span>}
                      {isDetectingAI ? 'Đang phân tích...' : 'Tự động sửa cấu hình bằng AI'}
                    </button>
                  </div>

                  <button
                    type="button"
                    className="ios-button ios-button-secondary"
                    onClick={handleAddBlurBox}
                  >
                    + Thêm vùng làm mờ tuỳ chỉnh
                  </button>
                  <p className="hint-text blur-hint">
                    Kéo thả trực tiếp vùng làm mờ trên màn hình Live View bên cạnh. Có thể kéo góc để phóng to/thu nhỏ. Nháy đúp chuột để xoá.
                  </p>
                </div>

                {form.cinematic_bars_enabled && (
                  <RangeField
                    label="Độ dày dải đen (trên/dưới)"
                    value={form.cinematic_bars_height_percent}
                    min={1}
                    max={40}
                    onChange={(value) => setField("cinematic_bars_height_percent", value)}
                    suffix="%"
                  />
                )}

                {form.blur_box_enabled && (
                  <div className="two-fields blur-options-grid">
                    <RangeField
                      label="Vị trí dọc thanh mờ"
                      value={form.blur_box_y_percent}
                      min={0}
                      max={100}
                      onChange={(value) => setField("blur_box_y_percent", value)}
                      suffix="%"
                    />
                    <RangeField
                      label="Độ cao thanh mờ"
                      value={form.blur_box_height_percent}
                      min={5}
                      max={100}
                      onChange={(value) => setField("blur_box_height_percent", value)}
                      suffix="%"
                    />
                  </div>
                )}

                {form.logo_enabled && (
                  <div className="logo-config">
                    <SectionTitle index="04" title="Cấu hình Logo" />
                    <div className="upload-row">
                      <button type="button" className="ios-button" onClick={() => document.getElementById('watermark-input')?.click()}>
                        {isUploading ? <Loader2 className="spin" size={17} /> : <Upload size={17} />}
                        <span>Chọn logo</span>
                      </button>
                      <input id="watermark-input" className="file-input-hidden" type="file" accept="image/*" onChange={handleWatermark} />
                      <span className="upload-name">
                        <FileImage size={17} />
                        {watermarkName}
                      </span>
                    </div>
                    <div className="two-fields">
                      <RangeField
                        label="Vị trí ngang (X)"
                        value={form.logo_x_percent}
                        min={0}
                        max={100}
                        onChange={(value) => setField("logo_x_percent", value)}
                        suffix="%"
                      />
                      <RangeField
                        label="Vị trí dọc (Y)"
                        value={form.logo_y_percent}
                        min={0}
                        max={100}
                        onChange={(value) => setField("logo_y_percent", value)}
                        suffix="%"
                      />
                    </div>
                    <RangeField
                      label="Kích thước rộng logo"
                      value={form.logo_width}
                      min={32}
                      max={800}
                      onChange={(value) => setField("logo_width", value)}
                      suffix="px"
                    />
                  </div>
                )}

                {message && <div className="form-message">{message}</div>}
              </section>
            </div>

            <div className="preview-column">
              <section className="ios-card">
                <SectionTitle index="05" title="Live view phụ đề" />
                <LivePreview 
                  subtitle_x_percent={form.subtitle_x_percent}
                  subtitle_y_percent={form.subtitle_y_percent}
                  subtitle_font_size={form.subtitle_font_size}
                  subtitle_box_enabled={form.subtitle_box_enabled}
                  subtitle_box_opacity={form.subtitle_box_opacity}
                  subtitle_box_height_percent={form.subtitle_box_height_percent}
                  hard_subtitles={form.hard_subtitles}
                  previewVideoUrl={previewVideoUrl}
                  logo_enabled={form.logo_enabled}
                  watermark_file_name={form.watermark_file_name ?? null}
                  logo_x_percent={form.logo_x_percent}
                  logo_y_percent={form.logo_y_percent}
                  logo_width={form.logo_width}
                  blur_box_enabled={form.blur_box_enabled}
                  blur_box_y_percent={form.blur_box_y_percent}
                  blur_box_height_percent={form.blur_box_height_percent}
                  custom_blur_boxes={form.custom_blur_boxes}
                  cinematic_bars_enabled={form.cinematic_bars_enabled}
                  cinematic_bars_height_percent={form.cinematic_bars_height_percent}
                  setField={setField} 
                />
                
                <div className="editor-grid" style={{ opacity: form.hard_subtitles ? 1 : 0.4, pointerEvents: form.hard_subtitles ? "auto" : "none" }}>
                  <RangeField label="Vị trí ngang" value={form.subtitle_x_percent} min={0} max={100} onChange={(value) => setField("subtitle_x_percent", value)} suffix="%" />
                  <RangeField label="Vị trí dọc" value={form.subtitle_y_percent} min={8} max={94} onChange={(value) => setField("subtitle_y_percent", value)} suffix="%" />
                  <RangeField label="Cỡ chữ" value={form.subtitle_font_size} min={16} max={120} onChange={(value) => setField("subtitle_font_size", value)} suffix="px" />
                  <RangeField
                    label="Cao thanh mờ"
                    value={form.subtitle_box_height_percent}
                    min={8}
                    max={45}
                    onChange={(value) => setField("subtitle_box_height_percent", value)}
                    suffix="%"
                  />
                  <RangeField
                    label="Độ mờ thanh"
                    value={form.subtitle_box_opacity}
                    min={0}
                    max={100}
                    onChange={(value) => setField("subtitle_box_opacity", value)}
                    suffix="%"
                  />
                  <CheckBox
                    checked={form.subtitle_box_enabled}
                    label="Bật thanh làm mờ sau phụ đề"
                    onChange={() => setField("subtitle_box_enabled", !form.subtitle_box_enabled)}
                  />
                </div>
              </section>

              <aside className="side-stack">
                <StatusPanel job={activeJob} />
                <HistoryPanel jobs={jobs} onSelect={handleSelectJob} onClear={handleClearJobs} isClearing={isClearingJobs} />
              </aside>
            </div>
          </form>
        </div>
      )}
    </main>
  );
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exponent;
  const digits = exponent === 0 ? 0 : value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[exponent]}`;
}

function formatUploadSpeed(bytesPerSecond: number): string {
  if (!Number.isFinite(bytesPerSecond) || bytesPerSecond <= 0) return "0 B/s";
  return `${formatBytes(bytesPerSecond)}/s`;
}

const SectionTitle = React.memo(function SectionTitle({ index, title }: { index: string; title: string }) {
  return (
    <div className="section-title">
      <span>{index}</span>
      <h2>{title}</h2>
    </div>
  );
});

interface RangeFieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  suffix: string;
  onChange: (value: number) => void;
}

const RangeField = React.memo(function RangeField({
  label,
  value,
  min,
  max,
  suffix,
  onChange,
}: RangeFieldProps) {
  const [localVal, setLocalVal] = useState(value);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    setLocalVal(value);
  }, [value]);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const num = Number(event.target.value);
    setLocalVal(num);

    if (timerRef.current) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => {
      onChange(num);
    }, 80);
  };

  return (
    <label className="range-field">
      <span>
        {label}
        <strong>
          {localVal}
          {suffix}
        </strong>
      </span>
      <input type="range" min={min} max={max} value={localVal} onChange={handleChange} />
    </label>
  );
});

interface SegmentedControlProps<T extends string> {
  label: string;
  value: T;
  options: Array<{ label: string; value: T }>;
  onChange: (value: T) => void;
}

const SegmentedControl = React.memo(function SegmentedControl<T extends string>({
  label,
  value,
  options,
  onChange,
}: SegmentedControlProps<T>) {
  return (
    <div className="segment-wrap">
      <span className="label">{label}</span>
      <div className="segments">
        {options.map((option) => (
          <button
            key={option.value}
            className={value === option.value ? "active" : ""}
            type="button"
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}) as <T extends string>(props: SegmentedControlProps<T>) => React.ReactElement;

const CheckBox = React.memo(function CheckBox({
  checked,
  label,
  onChange,
  disabled = false,
}: {
  checked: boolean;
  label: string;
  onChange: () => void;
  disabled?: boolean;
}) {
  return (
    <label className={`check-item ${disabled ? "disabled" : ""}`}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={onChange} />
      <span>{checked && <Check size={14} />}</span>
      {label}
    </label>
  );
});

const StatusPanel = React.memo(function StatusPanel({ job }: { job: JobProgress | null }) {
  const [etaText, setEtaText] = useState<string | null>(null);
  const lastProgressRef = useRef<{ progress: number; time: number; stage: string } | null>(null);

  useEffect(() => {
    if (!job || job.status !== "processing" || job.progress >= 100) {
      setEtaText(null);
      lastProgressRef.current = null;
      return;
    }

    const now = Date.now();
    const current = lastProgressRef.current;

    if (!current || job.stage !== current.stage) {
      lastProgressRef.current = { progress: job.progress, time: now, stage: job.stage };
      setEtaText(null);
    } else {
      const progressDelta = job.progress - current.progress;
      const timeDelta = now - current.time;

      if (progressDelta > 0) {
        const remainingProgress = 100 - job.progress;
        const timePerPercent = timeDelta / progressDelta;
        const msRemaining = timePerPercent * remainingProgress;

        const secs = Math.ceil(msRemaining / 1000);
        if (secs > 60) {
          setEtaText(`~ ${Math.floor(secs / 60)} phút ${secs % 60} giây`);
        } else {
          setEtaText(`~ ${secs} giây`);
        }
      }
    }
  }, [job]);

  const seoTags = useMemo(() => {
    return (job?.seo_tags ?? [])
      .map((tag) => tag.trim())
      .filter(Boolean)
      .map((tag) => (tag.startsWith("#") ? tag : `#${tag.replace(/^#+/, "").replace(/\s+/g, "")}`))
      .join(" ");
  }, [job?.seo_tags]);

  const youtubeUploadText = useMemo(() => {
    if (!job) return "";
    const description = job.seo_description ?? "";
    const shouldAppendTags = seoTags && !description.includes("#");
    return [job.seo_title, description, shouldAppendTags ? seoTags : ""].filter(Boolean).join("\n\n");
  }, [job, seoTags]);

  const copyToClipboard = useCallback(async (value: string) => {
    if (!value) return;
    await navigator.clipboard.writeText(value);
  }, []);

  if (!job) {
    return (
      <section className="ios-card">
        <h2>Trạng thái</h2>
        <p style={{ color: "#86868b", marginTop: "8px" }}>Chưa có job đang chọn.</p>
      </section>
    );
  }

  return (
    <section className="ios-card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <span style={{ fontSize: "0.85rem", fontWeight: 700, textTransform: "uppercase", padding: "4px 10px", borderRadius: "999px", background: "rgba(0,0,0,0.05)" }}>
            {statusLabel(job.status)}
          </span>
          {(job.status === "processing" || job.status === "queued") && (
            <button
              type="button"
              className="ios-button ios-button-secondary"
              style={{ padding: "4px 12px", fontSize: "0.8rem" }}
              onClick={async () => {
                if (window.confirm("Bạn có chắc chắn muốn hủy tiến trình này?")) {
                  try {
                    await cancelJob(job.job_id);
                  } catch (e) {
                    alert("Không thể hủy: " + (e as Error).message);
                  }
                }
              }}
            >
              Hủy xử lý
            </button>
          )}
        </div>
        <span>{job.progress}%</span>
      </div>
      <h2 style={{ fontSize: "1.1rem", marginBottom: "8px" }}>{job.stage}</h2>
      {etaText && (
        <p style={{ color: "#86868b", fontSize: "0.9rem", marginBottom: "16px" }}>
          Dự kiến còn: <strong>{etaText}</strong>
        </p>
      )}
      <div style={{ height: "12px", background: "rgba(0,0,0,0.05)", borderRadius: "999px", overflow: "hidden", marginBottom: "16px" }}>
        <div style={{ width: `${job.progress}%`, height: "100%", background: "#0071e3", borderRadius: "999px", transition: "width 0.3s" }} />
      </div>
      {job.error && <ErrorMessage error={job.error} />}
      {job.output_video_url && (
        <a className="ios-button" href={toAbsoluteApiUrl(job.output_video_url)} target="_blank" rel="noreferrer" style={{ textDecoration: "none", width: "100%" }}>
          <Download size={17} />
          Tải video đầu ra
        </a>
      )}
      {job.status === "completed" && youtubeUploadText && (
        <div style={{ marginTop: 16, display: "grid", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
            <h3 style={{ fontSize: "1rem", margin: 0 }}>Nội dung up YouTube</h3>
            <button type="button" className="ios-button ios-button-secondary" style={{ padding: "6px 10px", fontSize: "0.82rem" }} onClick={() => copyToClipboard(youtubeUploadText)}>
              <Copy size={15} />
              Copy tất cả
            </button>
          </div>
          {job.seo_title && <SeoCopyBlock label="Tiêu đề" value={job.seo_title} onCopy={copyToClipboard} />}
          {job.seo_description && <SeoCopyBlock label="Mô tả" value={job.seo_description} onCopy={copyToClipboard} multiline />}
          {seoTags && <SeoCopyBlock label="Hashtag" value={seoTags} onCopy={copyToClipboard} />}
        </div>
      )}
      {job.status === "completed" && job.created_at && job.updated_at && (
        <p className="muted" style={{ marginTop: 12, fontSize: "0.85rem" }}>
          Tổng thời gian xử lý:{" "}
          <strong>
            {(() => {
              const diffSecs = Math.floor((new Date(job.updated_at).getTime() - new Date(job.created_at).getTime()) / 1000);
              return diffSecs > 60 ? `${Math.floor(diffSecs / 60)} phút ${diffSecs % 60} giây` : `${diffSecs} giây`;
            })()}
          </strong>
        </p>
      )}
    </section>
  );
});

const SeoCopyBlock = React.memo(function SeoCopyBlock({
  label,
  value,
  onCopy,
  multiline = false,
}: {
  label: string;
  value: string;
  onCopy: (value: string) => void;
  multiline?: boolean;
}) {
  return (
    <div style={{ border: "1px solid rgba(0,0,0,0.08)", borderRadius: 12, padding: 12, background: "#f7f8fa" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, marginBottom: 8 }}>
        <strong style={{ fontSize: "0.88rem" }}>{label}</strong>
        <button type="button" className="ios-button ios-button-secondary" style={{ padding: "4px 8px", fontSize: "0.78rem" }} onClick={() => onCopy(value)}>
          <Copy size={14} />
          Copy
        </button>
      </div>
      <p style={{ whiteSpace: "pre-wrap", margin: 0, color: "#1d1d1f", fontSize: multiline ? "0.88rem" : "0.95rem", lineHeight: 1.45 }}>{value}</p>
    </div>
  );
});

interface HistoryPanelProps {
  jobs: JobProgress[];
  onSelect: (id: string) => void;
  onClear: () => void;
  isClearing: boolean;
}

const HistoryPanel = React.memo(function HistoryPanel({ jobs, onSelect, onClear, isClearing }: HistoryPanelProps) {
  return (
    <section className="ios-card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
        <h2 style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "1.2rem" }}>
          <Video size={20} color="#0071e3" />
          Job gần đây
        </h2>
        {jobs.length > 0 && (
          <button
            className="ios-button ios-button-secondary"
            type="button"
            onClick={onClear}
            disabled={isClearing}
            title="Đưa file job và video đã render vào Thùng rác"
            style={{ padding: "6px 12px", fontSize: "0.85rem", color: "#d32f2f" }}
          >
            {isClearing ? <Loader2 className="spin" size={17} /> : <Trash2 size={17} />}
            <span>Xoá tất cả</span>
          </button>
        )}
      </div>
      {jobs.length === 0 ? (
        <p style={{ color: "#86868b" }}>Chưa có job nào.</p>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          {jobs.map((job) => (
            <button
              key={job.job_id}
              type="button"
              onClick={() => onSelect(job.job_id)}
              style={{
                display: "flex",
                alignItems: "flex-start",
                gap: "12px",
                padding: "12px",
                background: "rgba(255,255,255,0.5)",
                border: "1px solid rgba(255,255,255,0.8)",
                borderRadius: "16px",
                textAlign: "left",
                width: "100%",
                transition: "all 0.2s",
              }}
            >
              <span
                style={{
                  width: "10px",
                  height: "10px",
                  borderRadius: "50%",
                  background: job.status === "completed" ? "#34c759" : job.status === "failed" ? "#ff3b30" : "#007aff",
                  marginTop: "6px",
                  flexShrink: 0,
                }}
              />
              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                <strong style={{ fontSize: "0.95rem" }}>{job.stage}</strong>
                <span style={{ fontSize: "0.85rem", color: "#86868b" }}>
                  {job.progress}% - {languageLabel(job.request.source_language)}
                  {job.status === "completed" && job.created_at && job.updated_at && (
                    <>
                      {" • "}
                      {(() => {
                        const diffSecs = Math.floor((new Date(job.updated_at).getTime() - new Date(job.created_at).getTime()) / 1000);
                        return diffSecs > 60 ? `${Math.floor(diffSecs / 60)} phút ${diffSecs % 60} giây` : `${diffSecs} giây`;
                      })()}
                    </>
                  )}
                </span>
              </div>
            </button>
          ))}
        </div>
      )}
    </section>
  );
});

function ErrorMessage({ error }: { error: string }) {
  return (
    <div style={{ background: "#fdecec", border: "1px solid #f8bbd0", padding: "16px", borderRadius: "16px", marginBottom: "16px", color: "#d32f2f" }}>
      <strong style={{ display: "block", marginBottom: "8px" }}>{friendlyError(error)}</strong>
      <details>
        <summary style={{ cursor: "pointer", fontWeight: 600, fontSize: "0.9rem" }}>Chi tiết kỹ thuật</summary>
        <pre style={{ marginTop: "8px", fontSize: "0.8rem", whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{error}</pre>
      </details>
    </div>
  );
}

function statusLabel(status: JobProgress["status"]) {
  const labels: Record<JobProgress["status"], string> = {
    queued: "Đang chờ",
    processing: "Đang xử lý",
    completed: "Hoàn tất",
    failed: "Thất bại",
  };
  return labels[status];
}

function languageLabel(value: string) {
  return languages.find((language) => language.value === value)?.label ?? value.toUpperCase();
}

function friendlyError(error: string) {
  const value = error.toLowerCase();
  const platform = value.includes("douyin") ? "Douyin" : value.includes("tiktok") ? "TikTok" : value.includes("youtube") ? "YouTube" : "nền tảng này";
  if (value.includes("edge tts") || value.includes("voice") || value.includes("tts") || value.includes("no audio was received")) {
    return "Không tạo được giọng đọc tiếng Việt. Backend sẽ tự chia nhỏ và thử lại, nếu vẫn lỗi hãy kiểm tra mạng/Edge TTS.";
  }
  if (value.includes("video unavailable") || value.includes("restricted")) {
    return "Video này đang bị nền tảng hoặc mạng/tài khoản hạn chế, backend không tải được. Hãy thử link công khai khác, dùng file nội bộ, hoặc cấu hình cookie hợp lệ.";
  }
  if (value.includes("fresh cookies") || value.includes("login") || value.includes("captcha") || value.includes("verify your identity") || value.includes("verification")) {
    return `${platform} yêu cầu cookie đăng nhập mới hoặc đang chặn xác minh. Hãy dùng link công khai khác, tải video về máy rồi chọn File nội bộ, hoặc xuất cookie Netscape và cấu hình AUTO_TRANSLATE_YTDLP_COOKIES_FILE.`;
  }
  if (value.includes("could not copy") || value.includes("cookie database") || value.includes("could not read")) {
    return "Không đọc được cookie trình duyệt. Hãy đóng Chrome/Edge rồi thử lại, hoặc xuất cookie ra file Netscape và cấu hình AUTO_TRANSLATE_YTDLP_COOKIES_FILE.";
  }
  if (value.includes("cookie")) {
    return `${platform} cần cookie hợp lệ. Hãy dùng link công khai khác, tải video về máy rồi chọn File nội bộ, hoặc cấu hình AUTO_TRANSLATE_YTDLP_COOKIES_FILE.`;
  }
  if (value.includes("ffmpeg")) {
    return "FFmpeg xử lý video thất bại. Kiểm tra file nguồn hoặc định dạng video.";
  }
  return error.split("\n")[0] || "Job xử lý thất bại.";
}
