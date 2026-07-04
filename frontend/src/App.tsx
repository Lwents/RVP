import { ChangeEvent, FormEvent, PointerEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  BadgeCheck,
  Check,
  ChevronRight,
  Download,
  FileImage,
  Loader2,
  Play,
  Trash2,
  Upload,
  Video,
  Wand2,
} from "lucide-react";
import { clearJobs, createJob, getJob, listJobs, toAbsoluteApiUrl, uploadVideo, uploadWatermark } from "./lib/api";
import type { BgmMode, DubbingRequest, JobProgress, LogoPosition, VoiceGender } from "./types/api";

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
  auto_publish: [],
  clone_voice: false,
  hard_subtitles: true,
  source_has_hard_subtitles: false,
  subtitle_x_percent: 50,
  subtitle_y_percent: 78,
  subtitle_font_size: 32,
  subtitle_box_enabled: true,
  subtitle_box_opacity: 55,
  subtitle_box_height_percent: 20,
  source_language: "auto",
  ducking_volume_db: -12,
  logo_position: "top_right",
  logo_width: 150,
  watermark_file_name: null,
};

export function App() {
  const [form, setForm] = useState<DubbingRequest>(defaultForm);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeJob, setActiveJob] = useState<JobProgress | null>(null);
  const [jobs, setJobs] = useState<JobProgress[]>([]);
  const [isSubmitting, setSubmitting] = useState(false);
  const [isUploading, setUploading] = useState(false);
  const [isVideoUploading, setVideoUploading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [watermarkName, setWatermarkName] = useState("Chưa có logo được tải lên");
  const [videoName, setVideoName] = useState("Chưa có video được import");

  const videoInputRef = useRef<HTMLInputElement | null>(null);

  const canSubmit = useMemo(() => {
    return Boolean(form.source_url?.trim() || form.local_file_path?.trim()) && !isSubmitting;
  }, [form.local_file_path, form.source_url, isSubmitting]);

  useEffect(() => {
    listJobs().then(setJobs).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!activeJobId) return;

    let cancelled = false;
    const poll = async () => {
      try {
        const job = await getJob(activeJobId);
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

  function setField<K extends keyof DubbingRequest>(key: K, value: DubbingRequest[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

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
    setMessage(null);
    try {
      const uploaded = await uploadVideo(file);
      if (!uploaded.local_file_path) {
        throw new Error("Backend không trả về đường dẫn video đã import.");
      }
      setForm((current) => ({
        ...current,
        source_url: "",
        local_file_path: uploaded.local_file_path,
      }));
      setVideoName(file.name);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Import video thất bại.");
    } finally {
      setVideoUploading(false);
      event.target.value = "";
    }
  }

  function openVideoPicker() {
    if (isVideoUploading) return;
    videoInputRef.current?.click();
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setMessage(null);

    const payload: DubbingRequest = {
      ...form,
      source_url: form.source_url?.trim() || null,
      local_file_path: form.local_file_path?.trim() || null,
    };

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

  async function handleClearJobs() {
    const confirmed = window.confirm("Đưa toàn bộ file job và video đã render vào Thùng rác, đồng thời xoá lịch sử job?");
    if (!confirmed) return;

    setMessage(null);
    try {
      await clearJobs();
      setJobs([]);
      setActiveJob(null);
      setActiveJobId(null);
      setMessage("Đã xoá lịch sử job.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Không thể xoá lịch sử job.");
    }
  }

  return (
    <main className="app-shell">
      <nav className="topbar">
        <a className="brand" href="/" aria-label="Auto-Translate AI">
          <span className="brand-mark">
            <Wand2 size={21} />
          </span>
          <span>Auto-Translate AI</span>
        </a>
        <div className="topbar-actions">
          <span className="health-pill">
            <BadgeCheck size={16} />
            Backend đang chạy
          </span>
          <a className="docs-link" href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer">
            Tài liệu API
            <ChevronRight size={16} />
          </a>
        </div>
      </nav>

      <header className="page-heading">
        <div>
          <p>Video dubbing workspace</p>
          <h1>Cấu hình, kéo vị trí phụ đề và render video trong một màn hình.</h1>
        </div>
        <button className="primary-action compact" type="submit" form="dubbing-form" disabled={!canSubmit}>
          {isSubmitting ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
          Render video
        </button>
      </header>

      <form id="dubbing-form" className="workspace" onSubmit={handleSubmit}>
        <section className="panel control-panel">
          <SectionTitle index="01" title="Nguồn video" />
          <label className="field">
            <span>URL video</span>
            <input
              value={form.source_url ?? ""}
              onChange={(event) => setField("source_url", event.target.value)}
              placeholder="YouTube, TikTok hoặc Douyin"
            />
          </label>
          <div className="field">
            <span>File nội bộ</span>
            <div className="upload-row file-import-row">
              <button className="upload-button" type="button" onClick={openVideoPicker} disabled={isVideoUploading}>
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
          </div>

          <div className="two-fields">
            <label className="field">
              <span>Ngôn ngữ gốc</span>
              <select value={form.source_language} onChange={(event) => setField("source_language", event.target.value as DubbingRequest["source_language"])}>
                {languages.map((language) => (
                  <option key={language.value} value={language.value}>
                    {language.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Ducking (dB)</span>
              <input
                type="number"
                min={-36}
                max={0}
                value={form.ducking_volume_db}
                onChange={(event) => setField("ducking_volume_db", Number(event.target.value))}
              />
            </label>
          </div>

          <SectionTitle index="02" title="Tuỳ chọn xử lý" />
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

          <div className="check-grid">
            <CheckBox checked={form.hard_subtitles} label="Ghi phụ đề vào video" onChange={() => setField("hard_subtitles", !form.hard_subtitles)} />
            <CheckBox
              checked={form.source_has_hard_subtitles}
              label="Video đã có phụ đề sẵn"
              onChange={() => setField("source_has_hard_subtitles", !form.source_has_hard_subtitles)}
            />
            <CheckBox checked={false} disabled label="Clone giọng thủ công - chờ voice engine" onChange={() => undefined} />
            <CheckBox checked={false} disabled label="YouTube Channel - chờ token" onChange={() => undefined} />
            <CheckBox checked={false} disabled label="Facebook Page - chờ token" onChange={() => undefined} />
          </div>

          <SectionTitle index="03" title="Logo" />
          <div className="upload-row">
            <label className="upload-button">
              {isUploading ? <Loader2 className="spin" size={17} /> : <Upload size={17} />}
              <span>Chọn logo</span>
              <input type="file" accept="image/*" onChange={handleWatermark} />
            </label>
            <span className="upload-name">
              <FileImage size={17} />
              {watermarkName}
            </span>
          </div>
          <div className="two-fields">
            <label className="field">
              <span>Vị trí logo</span>
              <select value={form.logo_position} onChange={(event) => setField("logo_position", event.target.value as LogoPosition)}>
                <option value="top_right">Trên phải</option>
                <option value="top_left">Trên trái</option>
                <option value="bottom_right">Dưới phải</option>
                <option value="bottom_left">Dưới trái</option>
              </select>
            </label>
            <label className="field">
              <span>Rộng logo</span>
              <input
                type="number"
                min={32}
                max={800}
                value={form.logo_width}
                onChange={(event) => setField("logo_width", Number(event.target.value))}
              />
            </label>
          </div>

          {message && <div className="message">{message}</div>}
        </section>

        <section className="panel preview-panel">
          <SectionTitle index="04" title="Live view phụ đề" />
          <LivePreview form={form} setField={setField} />
        </section>

        <aside className="side-stack">
          <StatusPanel job={activeJob} />
          <HistoryPanel jobs={jobs} onSelect={setActiveJobId} onClear={handleClearJobs} />
        </aside>
      </form>
    </main>
  );
}

function SectionTitle({ index, title }: { index: string; title: string }) {
  return (
    <div className="section-title">
      <span>{index}</span>
      <h2>{title}</h2>
    </div>
  );
}

interface LivePreviewProps {
  form: DubbingRequest;
  setField: <K extends keyof DubbingRequest>(key: K, value: DubbingRequest[K]) => void;
}

function LivePreview({ form, setField }: LivePreviewProps) {
  const frameRef = useRef<HTMLDivElement | null>(null);

  function updatePosition(event: PointerEvent<HTMLDivElement>) {
    const frame = frameRef.current;
    if (!frame) return;
    const rect = frame.getBoundingClientRect();
    const x = Math.round(((event.clientX - rect.left) / rect.width) * 100);
    const y = Math.round(((event.clientY - rect.top) / rect.height) * 100);
    setField("subtitle_x_percent", clamp(x, 5, 95));
    setField("subtitle_y_percent", clamp(y, 8, 94));
  }

  function beginDrag(event: PointerEvent<HTMLDivElement>) {
    event.currentTarget.setPointerCapture(event.pointerId);
    updatePosition(event);
  }

  const boxTop = clamp(form.subtitle_y_percent - form.subtitle_box_height_percent / 2, 0, 100 - form.subtitle_box_height_percent);

  return (
    <div className="live-editor">
      <div className="video-frame" ref={frameRef}>
        <div className="sample-scene">
          <div className="scene-grid" />
          <div className="scene-person" />
          <div className="scene-caption">Preview frame</div>
        </div>
        {form.subtitle_box_enabled && (
          <div
            className="blur-band"
            style={{
              top: `${boxTop}%`,
              height: `${form.subtitle_box_height_percent}%`,
              opacity: form.subtitle_box_opacity / 100,
            }}
          />
        )}
        <div
          className="subtitle-layer"
          style={{
            left: `${form.subtitle_x_percent}%`,
            top: `${form.subtitle_y_percent}%`,
            fontSize: `${Math.max(16, form.subtitle_font_size * 0.72)}px`,
          }}
          onPointerDown={beginDrag}
          onPointerMove={(event) => {
            if (event.buttons === 1) updatePosition(event);
          }}
        >
          <span>Phụ đề sẽ nằm ở đây sau khi render</span>
        </div>
      </div>

      <div className="editor-grid">
        <RangeField label="Vị trí ngang" value={form.subtitle_x_percent} min={0} max={100} onChange={(value) => setField("subtitle_x_percent", value)} suffix="%" />
        <RangeField label="Vị trí dọc" value={form.subtitle_y_percent} min={8} max={94} onChange={(value) => setField("subtitle_y_percent", value)} suffix="%" />
        <RangeField label="Cỡ chữ" value={form.subtitle_font_size} min={16} max={72} onChange={(value) => setField("subtitle_font_size", value)} suffix="px" />
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
    </div>
  );
}

function RangeField({
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
        <strong>
          {value}
          {suffix}
        </strong>
      </span>
      <input type="range" min={min} max={max} value={value} onChange={(event) => onChange(Number(event.target.value))} />
    </label>
  );
}

interface SegmentedControlProps<T extends string> {
  label: string;
  value: T;
  options: Array<{ label: string; value: T }>;
  onChange: (value: T) => void;
}

function SegmentedControl<T extends string>({ label, value, options, onChange }: SegmentedControlProps<T>) {
  return (
    <div className="segment-wrap">
      <span className="label">{label}</span>
      <div className="segments">
        {options.map((option) => (
          <button key={option.value} className={value === option.value ? "active" : ""} type="button" onClick={() => onChange(option.value)}>
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function CheckBox({ checked, label, onChange, disabled = false }: { checked: boolean; label: string; onChange: () => void; disabled?: boolean }) {
  return (
    <label className={`check-item ${disabled ? "disabled" : ""}`}>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={onChange} />
      <span>{checked && <Check size={14} />}</span>
      {label}
    </label>
  );
}

function StatusPanel({ job }: { job: JobProgress | null }) {
  if (!job) {
    return (
      <section className="panel status-panel">
        <h2>Trạng thái</h2>
        <p className="muted">Chưa có job đang chọn.</p>
      </section>
    );
  }

  return (
    <section className="panel status-panel">
      <div className="status-heading">
        <span className={`status-pill ${job.status}`}>{statusLabel(job.status)}</span>
        <span>{job.progress}%</span>
      </div>
      <h2>{job.stage}</h2>
      <div className="progress-bar">
        <span style={{ width: `${job.progress}%` }} />
      </div>
      {job.error && <ErrorMessage error={job.error} />}
      {job.output_video_url && (
        <a className="download-link" href={toAbsoluteApiUrl(job.output_video_url)} target="_blank" rel="noreferrer">
          <Download size={17} />
          Tải video đầu ra
        </a>
      )}
    </section>
  );
}

function HistoryPanel({ jobs, onSelect, onClear }: { jobs: JobProgress[]; onSelect: (id: string) => void; onClear: () => void }) {
  return (
    <section className="panel history-panel">
      <div className="history-heading">
      <h2>
        <Video size={20} />
        Job gần đây
      </h2>
      {jobs.length > 0 && (
        <button className="trash-action" type="button" onClick={onClear} title="Đưa file job và video đã render vào Thùng rác">
          <Trash2 size={17} />
          <span>Xoá tất cả</span>
        </button>
      )}
      </div>
      {jobs.length === 0 ? (
        <p className="muted">Chưa có job nào.</p>
      ) : (
        <div className="job-list">
          {jobs.map((job) => (
            <button key={job.job_id} type="button" onClick={() => onSelect(job.job_id)}>
              <span className={`dot ${job.status}`} />
              <span>
                <strong>{job.stage}</strong>
                <small>
                  {job.progress}% - {languageLabel(job.request.source_language)}
                </small>
              </span>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

function ErrorMessage({ error }: { error: string }) {
  return (
    <div className="message error">
      <strong>{friendlyError(error)}</strong>
      <details>
        <summary>Chi tiết kỹ thuật</summary>
        <pre>{error}</pre>
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
  if (value.includes("video unavailable") || value.includes("restricted")) {
    return "Video này đang bị nền tảng hoặc mạng/tài khoản hạn chế, backend không tải được. Hãy thử link công khai khác, dùng file nội bộ, hoặc cấu hình cookie hợp lệ.";
  }
  if (value.includes("fresh cookies") || value.includes("login") || value.includes("captcha") || value.includes("verify")) {
    return `${platform} yêu cầu cookie đăng nhập mới hoặc đang chặn xác minh. Hãy dùng link công khai khác, tải video về máy rồi chọn File nội bộ, hoặc xuất cookie Netscape và cấu hình AUTO_TRANSLATE_YTDLP_COOKIES_FILE.`;
  }
  if (value.includes("could not copy") || value.includes("cookie database") || value.includes("could not read")) {
    return "Không đọc được cookie trình duyệt. Hãy đóng Chrome/Edge rồi thử lại, hoặc xuất cookie ra file Netscape và cấu hình AUTO_TRANSLATE_YTDLP_COOKIES_FILE.";
  }
  if (value.includes("cookie")) {
    return `${platform} cần cookie hợp lệ. Hãy dùng link công khai khác, tải video về máy rồi chọn File nội bộ, hoặc cấu hình AUTO_TRANSLATE_YTDLP_COOKIES_FILE.`;
  }
  if (value.includes("voice") || value.includes("tts")) {
    return "Không tạo được giọng đọc. Kiểm tra mạng và voice engine Edge TTS.";
  }
  if (value.includes("ffmpeg")) {
    return "FFmpeg xử lý video thất bại. Kiểm tra file nguồn hoặc định dạng video.";
  }
  return error.split("\n")[0] || "Job xử lý thất bại.";
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}
