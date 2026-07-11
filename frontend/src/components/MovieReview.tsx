import React, { ChangeEvent, useCallback, useEffect, useRef, useState } from "react";
import { Clapperboard, Copy, Download, Loader2, Sparkles, Trash2, Upload, Video } from "lucide-react";
import { clearReviewDraftJobs, createReviewDraftJob, getReviewDraftJob, listReviewDraftJobs, toAbsoluteApiUrl, uploadVideo } from "../lib/api";
import type { UploadProgress } from "../lib/api";
import type { ReviewBeat, ReviewDraftJob, ReviewDraftRequest, SourceLanguage } from "../types/api";

const reviewStyles: Array<{ label: string; value: ReviewDraftRequest["style"] }> = [
  { label: "Kể chuyện", value: "story" },
  { label: "Nhanh gọn", value: "fast" },
  { label: "Cảm xúc", value: "emotional" },
  { label: "Duyên hài", value: "funny" },
];

const sourceLanguages: Array<{ label: string; value: SourceLanguage }> = [
  { label: "Tự nhận diện", value: "auto" },
  { label: "Tiếng Trung", value: "zh" },
  { label: "Tiếng Anh", value: "en" },
  { label: "Tiếng Việt", value: "vi" },
];

export const MovieReview = React.memo(function MovieReview() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const previewRef = useRef<HTMLVideoElement | null>(null);
  const [videoPath, setVideoPath] = useState("");
  const [sourceVideoUrl, setSourceVideoUrl] = useState<string | null>(null);
  const [videoName, setVideoName] = useState("Chưa có phim được import");
  const [targetMinutes, setTargetMinutes] = useState(8);
  const [style, setStyle] = useState<ReviewDraftRequest["style"]>("story");
  const [sourceLanguage, setSourceLanguage] = useState<SourceLanguage>("auto");
  const [notes, setNotes] = useState("");
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const [isUploading, setUploading] = useState(false);
  const [isCreating, setCreating] = useState(false);
  const [isClearingHistory, setClearingHistory] = useState(false);
  const [job, setJob] = useState<ReviewDraftJob | null>(null);
  const [reviewJobs, setReviewJobs] = useState<ReviewDraftJob[]>([]);
  const [previewMode, setPreviewMode] = useState<"source" | "output">("source");
  const [selectedBeatIndex, setSelectedBeatIndex] = useState(0);
  const [message, setMessage] = useState<string | null>(null);

  const refreshReviewJobs = useCallback(async () => {
    const items = await listReviewDraftJobs();
    setReviewJobs(items);
    return items;
  }, []);

  useEffect(() => {
    let cancelled = false;
    refreshReviewJobs().then((items) => {
      if (!cancelled && items.length > 0) {
        setJob((current) => {
          if (current) return current;
          const item = items[0];
          setVideoPath(item.request.video_path);
          setSourceVideoUrl(sourceUrlFromPath(item.request.video_path));
          setVideoName(fileNameFromPath(item.request.video_path));
          setPreviewMode(item.result?.output_video_url ? "output" : "source");
          return item;
        });
      }
    }).catch((error) => {
      setMessage(error instanceof Error ? error.message : "Không thể tải lịch sử review job.");
    });
    return () => {
      cancelled = true;
    };
  }, [refreshReviewJobs]);

  useEffect(() => {
    if (!job || job.status === "completed" || job.status === "failed") return;

    let cancelled = false;
    const timer = window.setInterval(async () => {
      try {
        const nextJob = await getReviewDraftJob(job.job_id);
        if (!cancelled) {
          setJob(nextJob);
          setReviewJobs((items) => items.map((item) => item.job_id === nextJob.job_id ? nextJob : item));
        }
      } catch (error) {
        if (!cancelled) setMessage(error instanceof Error ? error.message : "Không thể lấy trạng thái review job.");
      }
    }, 1800);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [job]);

  const openPicker = useCallback(() => {
    if (!isUploading) inputRef.current?.click();
  }, [isUploading]);

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
      const created = await createReviewDraftJob({
        video_path: videoPath,
        target_minutes: targetMinutes,
        style,
        source_language: sourceLanguage,
        notes: notes.trim() || null,
      });
      setJob(created);
      setSelectedBeatIndex(0);
      await refreshReviewJobs();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Không thể tạo review job.");
    } finally {
      setCreating(false);
    }
  }, [notes, refreshReviewJobs, sourceLanguage, style, targetMinutes, videoPath]);

  const selectReviewJob = useCallback((item: ReviewDraftJob) => {
    setJob(item);
    setVideoPath(item.request.video_path);
    setSourceVideoUrl(sourceUrlFromPath(item.request.video_path));
    setVideoName(fileNameFromPath(item.request.video_path));
    setPreviewMode(item.result?.output_video_url ? "output" : "source");
    setSelectedBeatIndex(0);
  }, []);

  const result = job?.result ?? null;
  const outputVideoUrl = result?.output_video_url ? toAbsoluteApiUrl(result.output_video_url) : null;
  const activePreviewUrl = previewMode === "output" ? outputVideoUrl : sourceVideoUrl;
  const selectedBeat = result?.beats[selectedBeatIndex] ?? null;

  const jumpToBeat = useCallback((index: number) => {
    const beat = result?.beats[index];
    if (!beat || !previewRef.current) return;
    setSelectedBeatIndex(index);
    if (previewMode !== "source") {
      setPreviewMode("source");
      window.setTimeout(() => {
        if (previewRef.current && beat.start_seconds != null) {
          previewRef.current.currentTime = Math.max(0, beat.start_seconds);
          previewRef.current.play().catch(() => undefined);
        }
      }, 80);
      return;
    }
    if (beat.start_seconds != null) {
      previewRef.current.currentTime = Math.max(0, beat.start_seconds);
      previewRef.current.play().catch(() => undefined);
    }
  }, [previewMode, result]);

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
      setMessage(result.message || "Đã xóa lịch sử review và đưa file vào Thùng rác.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Không thể xóa lịch sử review.");
    } finally {
      setClearingHistory(false);
    }
  }, []);

  const copyText = useCallback(async (value: string) => {
    if (!value) return;
    await navigator.clipboard.writeText(value);
  }, []);

  return (
    <div className="ios-view-transition">
      <header className="page-heading">
        <div>
          <h1>Tạo review phim</h1>
          <p>Nạp phim dài, AI tạo kịch bản kể chuyện và danh sách cảnh gợi ý để dựng review.</p>
        </div>
        <button className="ios-button" type="button" onClick={createDraft} disabled={isUploading || isCreating || !videoPath}>
          {isCreating || (job && job.status !== "completed" && job.status !== "failed") ? <Loader2 className="spin" size={18} /> : <Sparkles size={18} />}
          Tạo bản review
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
              <span>Thời lượng review</span>
              <input className="ios-input" type="number" min={1} max={30} value={targetMinutes} onChange={(event) => setTargetMinutes(Number(event.target.value))} />
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

          <div className="segment-wrap">
            <span className="label">Phong cách review</span>
            <div className="segments">
              {reviewStyles.map((item) => (
                <button key={item.value} type="button" className={style === item.value ? "active" : ""} onClick={() => setStyle(item.value)}>
                  {item.label}
                </button>
              ))}
            </div>
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
                    className="review-beat"
                    onClick={() => selectReviewJob(item)}
                    style={{ textAlign: "left", cursor: "pointer" }}
                  >
                    <strong>{item.stage}</strong>
                    <span>{item.progress}% • {item.status} • {formatDateTime(item.created_at)}</span>
                    <p>{fileNameFromPath(item.request.video_path)} • {item.request.target_minutes} phút</p>
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
              {job.error && <div className="review-error">{job.error}</div>}
            </div>
          )}

          {result && (
            <div className="review-result">
              {result.output_video_url && (
                <a className="ios-button review-download" href={toAbsoluteApiUrl(result.output_video_url)} target="_blank" rel="noreferrer">
                  <Download size={17} />
                  Tải video review {result.target_minutes} phút
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
                  {result.beats.map((beat, index) => (
                    <button key={`${beat.time_hint}-${index}`} type="button" className={`review-beat ${index === selectedBeatIndex ? "active" : ""}`} onClick={() => jumpToBeat(index)}>
                      <strong>{beat.time_hint}</strong>
                      <span>{beat.purpose}</span>
                      <p>{beat.narration}</p>
                    </button>
                  ))}
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

const ReviewLivePreview = React.forwardRef<HTMLVideoElement, {
  mode: "source" | "output";
  onModeChange: (mode: "source" | "output") => void;
  sourceVideoUrl: string | null;
  outputVideoUrl: string | null;
  activeVideoUrl: string | null;
  selectedBeat: ReviewBeat | null;
}>(function ReviewLivePreview({ mode, onModeChange, sourceVideoUrl, outputVideoUrl, activeVideoUrl, selectedBeat }, ref) {
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
      <div className="video-frame review-live-frame">
        {activeVideoUrl ? (
          <video ref={ref} className="preview-video-element" src={activeVideoUrl} controls playsInline />
        ) : (
          <div className="review-live-empty">
            <Video size={36} />
            <strong>Import phim để xem live view</strong>
            <span>Khi có bản review, bạn có thể đổi qua tab Video review.</span>
          </div>
        )}
      </div>
      <div className="review-preview-meta">
        <strong>{mode === "output" ? "Đang xem video review đã render" : "Đang xem phim gốc để so cảnh"}</strong>
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

function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString("vi-VN", { hour: "2-digit", minute: "2-digit", day: "2-digit", month: "2-digit" });
}
