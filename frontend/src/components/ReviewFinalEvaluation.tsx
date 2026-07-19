import {
  AlertTriangle,
  Award,
  CheckCircle2,
  CircleAlert,
  Lightbulb,
  Sparkles,
  ThumbsUp,
} from "lucide-react";
import type {
  ReviewFinalEvaluation,
  ReviewFinalEvaluationCriterionKey,
  ReviewFinalEvaluationVerdict,
} from "../types/api";
import "./ReviewFinalEvaluation.css";

const verdictCopy: Record<ReviewFinalEvaluationVerdict, { label: string; description: string }> = {
  excellent: {
    label: "Xuất sắc",
    description: "Video đạt chất lượng rất cao theo bộ tiêu chí đánh giá cuối.",
  },
  good: {
    label: "Tốt",
    description: "Video đạt chất lượng tốt và chỉ còn ít điểm có thể tinh chỉnh.",
  },
  needs_improvement: {
    label: "Cần cải thiện",
    description: "Video còn một số điểm nên sửa trước khi xuất bản.",
  },
  poor: {
    label: "Chưa đạt",
    description: "Video cần được kiểm tra và chỉnh sửa lại trước khi sử dụng.",
  },
};

const criterionFallbackLabels: Record<ReviewFinalEvaluationCriterionKey, string> = {
  content_fidelity: "Đúng nội dung gốc",
  translation_accuracy: "Độ chính xác bản dịch",
  av_subtitle_sync: "Đồng bộ hình, tiếng và phụ đề",
  narrative_coherence: "Mạch kể chuyện",
  technical_quality: "Chất lượng kỹ thuật",
  safety_compliance: "An toàn nội dung",
};

export function ReviewFinalEvaluationPanel({ evaluation }: { evaluation: ReviewFinalEvaluation }) {
  const score = normalizeScore(evaluation.overall_score);
  const verdict = verdictCopy[evaluation.verdict] ?? verdictCopy.needs_improvement;
  const criteria = Array.isArray(evaluation.criteria) ? evaluation.criteria : [];
  const strengths = cleanList(evaluation.strengths);
  const recommendations = cleanList(evaluation.recommendations);

  return (
    <section
      className={`review-final-evaluation verdict-${evaluation.verdict} ${evaluation.passed ? "passed" : "not-passed"}`}
      aria-label="Kết quả AI đánh giá cuối cùng của video đã render"
    >
      <header className="review-final-head">
        <div className="review-final-heading">
          <span className="review-final-eyebrow"><Sparkles size={15} />AI đánh giá cuối cùng</span>
          <div className="review-final-title-row">
            <h3>{verdict.label}</h3>
            <span className={`review-final-pass-badge ${evaluation.passed ? "passed" : "not-passed"}`}>
              {evaluation.passed ? <CheckCircle2 size={14} /> : <CircleAlert size={14} />}
              {evaluation.passed ? "Đạt chuẩn" : "Cần kiểm tra"}
            </span>
          </div>
          <p>{evaluation.summary?.trim() || verdict.description}</p>
        </div>

        <div
          className="review-final-score-ring"
          style={{ background: `conic-gradient(var(--review-final-accent) ${score}%, rgba(142, 142, 147, 0.18) 0)` }}
          role="img"
          aria-label={`Tổng điểm ${score} trên 100`}
        >
          <span>{score}<small>/100</small></span>
        </div>
      </header>

      <p className="review-final-verdict-note">{verdict.description}</p>

      {criteria.length > 0 && (
        <div className="review-final-criteria" aria-label="Điểm theo từng tiêu chí">
          {criteria.map((criterion) => {
            const criterionScore = normalizeScore(criterion.score);
            const criterionLabel = criterion.label?.trim() || criterionFallbackLabels[criterion.key] || criterion.key;
            const findings = cleanList(criterion.findings);

            return (
              <article className={`review-final-criterion ${criterion.passed ? "passed" : "not-passed"}`} key={criterion.key}>
                <div className="review-final-criterion-head">
                  <div>
                    <strong>{criterionLabel}</strong>
                    <small>Trọng số {normalizeWeight(criterion.weight_percent)}%</small>
                  </div>
                  <span>{criterionScore}<small>/100</small></span>
                </div>
                <div
                  className="review-final-criterion-track"
                  role="progressbar"
                  aria-label={`${criterionLabel}: ${criterionScore} trên 100`}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={criterionScore}
                >
                  <i style={{ width: `${criterionScore}%` }} />
                </div>
                {criterion.feedback?.trim() && <p>{criterion.feedback.trim()}</p>}
                {findings.length > 0 && (
                  <details className="review-final-findings">
                    <summary>{findings.length} nhận xét chi tiết</summary>
                    <ul>
                      {findings.map((finding, index) => <li key={`${criterion.key}-finding-${index}`}>{finding}</li>)}
                    </ul>
                  </details>
                )}
              </article>
            );
          })}
        </div>
      )}

      {(strengths.length > 0 || recommendations.length > 0) && (
        <div className="review-final-insights">
          {strengths.length > 0 && (
            <div className="review-final-insight strengths">
              <h4><ThumbsUp size={16} />Điểm làm tốt</h4>
              <ul>{strengths.map((item, index) => <li key={`strength-${index}`}>{item}</li>)}</ul>
            </div>
          )}
          {recommendations.length > 0 && (
            <div className="review-final-insight recommendations">
              <h4><Lightbulb size={17} />Nên cải thiện</h4>
              <ul>{recommendations.map((item, index) => <li key={`recommendation-${index}`}>{item}</li>)}</ul>
            </div>
          )}
        </div>
      )}

      {evaluation.fallback_used && (
        <div className="review-final-fallback" role="note">
          <AlertTriangle size={16} />
          <span>AI hoặc dữ liệu đánh giá chưa đầy đủ nên hệ thống đã dùng bộ chấm dự phòng. Hãy xem kỹ các khuyến nghị trước khi đăng.</span>
        </div>
      )}

      <footer className="review-final-meta">
        <span><Award size={14} />Model: <strong>{evaluation.model?.trim() || "Không xác định"}</strong></span>
        {evaluation.evaluated_at && <time dateTime={evaluation.evaluated_at}>Đánh giá lúc {formatEvaluationDate(evaluation.evaluated_at)}</time>}
      </footer>
    </section>
  );
}

function normalizeScore(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

function normalizeWeight(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

function cleanList(items: string[] | null | undefined): string[] {
  if (!Array.isArray(items)) return [];
  return items.map((item) => item?.trim()).filter((item): item is string => Boolean(item));
}

function formatEvaluationDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("vi-VN", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
