from __future__ import annotations

import base64
import json
import math
import re
import unicodedata
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path

from openai import AsyncOpenAI

from app.core import settings
from app.services.ai.openai_client import get_async_openai
from app.models.review import (
    FinalReviewCriterion,
    FinalReviewCriterionKey,
    FinalReviewEvaluation,
    VerifiedReviewPackage,
)
from app.services.media.ffmpeg import find_ffmpeg, probe_video_duration
from app.services.subtitles.timing import SubtitleEvent, parse_srt


_CRITERIA: tuple[tuple[FinalReviewCriterionKey, str, float], ...] = (
    ("content_fidelity", "Độ đúng và bám sát nội dung", 25.0),
    ("translation_accuracy", "Chất lượng chuyển ngữ", 15.0),
    ("av_subtitle_sync", "Đồng bộ hình, giọng đọc và phụ đề", 20.0),
    ("narrative_coherence", "Mạch kể và tính liền mạch", 15.0),
    ("technical_quality", "Chất lượng kỹ thuật", 15.0),
    ("safety_compliance", "An toàn và phù hợp nội dung", 10.0),
)
_CRITERION_KEYS = {key for key, _, _ in _CRITERIA}
_CRITERION_PASS_SCORE = {
    "content_fidelity": 75.0,
    "translation_accuracy": 70.0,
    "av_subtitle_sync": 75.0,
    "narrative_coherence": 70.0,
    "technical_quality": 70.0,
    "safety_compliance": 80.0,
}
_AI_INFLUENCE = {
    "content_fidelity": 0.65,
    "translation_accuracy": 0.75,
    # Timing and media health have authoritative deterministic measurements.
    "av_subtitle_sync": 0.20,
    "narrative_coherence": 0.65,
    "technical_quality": 0.25,
    "safety_compliance": 1.00,
}


async def evaluate_final_review_video(
    rendered_video: Path,
    package: VerifiedReviewPackage,
    work_dir: Path,
    *,
    review_subtitle_file: Path | None = None,
    source_transcript_file: Path | None = None,
    word_timing_file: Path | None = None,
) -> FinalReviewEvaluation:
    """Score the actual final render without making publishing depend on AI uptime.

    The existing post-render gate remains authoritative for correctness. This
    evaluator is an independent, user-facing scorecard. A network/model error
    produces a deterministic report from the post-render QA, media duration,
    SRT and TTS word timings instead of failing an otherwise valid render.
    """

    work_dir.mkdir(parents=True, exist_ok=True)
    subtitle_file = review_subtitle_file if review_subtitle_file and review_subtitle_file.is_file() else None
    timing_file = word_timing_file or work_dir / "review_word_timings.json"
    signal_fallback_used = False
    try:
        signals = await _collect_final_signals(
            rendered_video,
            package,
            subtitle_file,
            timing_file if timing_file.is_file() else None,
        )
    except Exception as exc:
        print(f"Final review signal fallback: {exc}")
        signals = _default_signals(rendered_video, package)
        signal_fallback_used = True
    baselines = _baseline_scores(package, signals)
    ai_payload: dict = {}
    fallback_used = signal_fallback_used
    try:
        ai_payload = await _ask_final_judge(
            rendered_video,
            package,
            work_dir,
            signals,
            subtitle_file,
            source_transcript_file,
        )
        if not isinstance(ai_payload.get("criteria"), (dict, list)):
            fallback_used = True
    except Exception as exc:
        print(f"Final review evaluator fallback: {exc}")
        fallback_used = True
        ai_payload = {}

    evaluation = _build_evaluation(
        ai_payload,
        baselines,
        signals,
        fallback_used=fallback_used,
    )
    try:
        (work_dir / "final_evaluation.json").write_text(
            evaluation.model_dump_json(indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        # The scorecard is advisory: a transient artifact write problem must
        # never discard a final MP4 that already passed post-render QA.
        print(f"Cannot persist final review evaluation: {exc}")
    return evaluation


async def _collect_final_signals(
    rendered_video: Path,
    package: VerifiedReviewPackage,
    subtitle_file: Path | None,
    word_timing_file: Path | None,
) -> dict:
    ffmpeg = find_ffmpeg() or "ffmpeg"
    media_duration = await probe_video_duration(ffmpeg, rendered_video)
    expected_voice_duration = max(
        (item.voice_end for item in package.edit_decision_list),
        default=0.0,
    )
    duration_drift = (
        abs(media_duration - expected_voice_duration)
        if media_duration > 0 and expected_voice_duration > 0
        else max(media_duration, expected_voice_duration)
    )
    subtitle_events = _read_subtitles(subtitle_file)
    word_timings = _read_word_timings(
        word_timing_file,
        expected_voice_duration or media_duration,
    )
    subtitle_metrics = _subtitle_sync_metrics(
        subtitle_events,
        word_timings,
        package,
        media_duration,
    )
    issue_codes = [item.code for item in package.quality_report.issues]
    black_or_missing = sum(
        code in {"BLACK_FRAME", "NO_CLIP"} or "BLACK" in code
        for code in issue_codes
    )
    technical_duration_score = _clamp(100.0 - max(0.0, duration_drift - 0.10) * 35.0)
    technical_health_score = _clamp(100.0 - black_or_missing * 25.0)
    return {
        "media_duration_seconds": round(media_duration, 3),
        "expected_voice_duration_seconds": round(expected_voice_duration, 3),
        "voice_video_drift_seconds": round(duration_drift, 3),
        "video_file_size_bytes": rendered_video.stat().st_size if rendered_video.is_file() else 0,
        "post_render_passed": package.quality_report.passed,
        "post_render_issue_codes": issue_codes,
        "technical_duration_score": round(technical_duration_score, 2),
        "technical_health_score": round(technical_health_score, 2),
        "signal_collection_failed": False,
        **subtitle_metrics,
    }


def _default_signals(
    rendered_video: Path,
    package: VerifiedReviewPackage,
) -> dict:
    expected_voice_duration = max(
        (item.voice_end for item in package.edit_decision_list),
        default=0.0,
    )
    try:
        file_size = rendered_video.stat().st_size
    except OSError:
        file_size = 0
    return {
        "media_duration_seconds": 0.0,
        "expected_voice_duration_seconds": round(expected_voice_duration, 3),
        "voice_video_drift_seconds": None,
        "video_file_size_bytes": file_size,
        "post_render_passed": package.quality_report.passed,
        "post_render_issue_codes": [item.code for item in package.quality_report.issues],
        "technical_duration_score": min(package.quality_report.duration_adherence_score, 70.0),
        "technical_health_score": 70.0 if package.quality_report.passed else 50.0,
        "subtitle_cue_count": 0,
        "subtitle_text_match_percent": 0.0,
        "subtitle_sync_score": 50.0,
        "subtitle_mean_offset_seconds": None,
        "subtitle_max_offset_seconds": None,
        "subtitle_out_of_bounds_cues": 0,
        "subtitle_timeline_coverage_percent": 0.0,
        "subtitle_sync_source": "unknown",
        "signal_collection_failed": True,
    }


def _read_subtitles(path: Path | None) -> list[SubtitleEvent]:
    if path is None or not path.is_file():
        return []
    try:
        return parse_srt(path)
    except Exception:
        return []


def _read_word_timings(
    path: Path | None,
    target_duration: float | None = None,
) -> list[dict]:
    if path is None or not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return []
    source_duration = 0.0
    if isinstance(data, dict):
        try:
            source_duration = float(data.get("audio_duration") or 0.0)
        except (TypeError, ValueError):
            source_duration = 0.0
        for key in ("words", "timings", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        return []
    scale = (
        float(target_duration) / source_duration
        if target_duration is not None
        and target_duration > 0
        and source_duration > 0
        else 1.0
    )
    result: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start") or 0.0) * scale
            end = float(item.get("end") or 0.0) * scale
        except (TypeError, ValueError):
            continue
        text = str(item.get("text") or item.get("word") or "").strip()
        if target_duration is not None and target_duration > 0:
            start = min(max(0.0, start), target_duration)
            end = min(max(start + 0.01, end), target_duration)
        if text and end > start:
            result.append({"text": text, "start": start, "end": end})
    return result


def _subtitle_sync_metrics(
    events: list[SubtitleEvent],
    word_timings: list[dict],
    package: VerifiedReviewPackage,
    media_duration: float,
) -> dict:
    if not events:
        return {
            "subtitle_cue_count": 0,
            "subtitle_text_match_percent": 0.0,
            "subtitle_sync_score": 0.0,
            "subtitle_mean_offset_seconds": None,
            "subtitle_max_offset_seconds": None,
            "subtitle_out_of_bounds_cues": 0,
            "subtitle_timeline_coverage_percent": 0.0,
            "subtitle_sync_source": "missing",
        }

    subtitle_text = " ".join(item.text for item in events)
    subtitle_key = _alignment_key(subtitle_text)
    narration_key = _alignment_key(package.narration_script)
    text_match = (
        SequenceMatcher(None, subtitle_key, narration_key).ratio() * 100.0
        if subtitle_key and narration_key
        else 0.0
    )
    out_of_bounds = sum(
        item.start < -0.01
        or item.end <= item.start
        or (media_duration > 0 and item.end > media_duration + 0.15)
        for item in events
    )
    timeline_duration = max(media_duration, max((item.end for item in events), default=0.0), 0.01)
    visible_duration = sum(max(0.0, item.end - item.start) for item in events)
    timeline_coverage = min(100.0, 100.0 * visible_duration / timeline_duration)

    offsets = _word_timing_offsets(events, word_timings, media_duration)
    sync_source = "word_timings"
    if offsets is None:
        offsets, range_score = _voice_range_offsets(events, package)
        sync_source = "voice_ranges"
    else:
        mean_offset = sum(offsets) / max(len(offsets), 1)
        max_offset = max(offsets, default=0.0)
        # 80 ms early / 180 ms late padding is intentional. The first 120 ms
        # of residual error is therefore free; larger offsets fall quickly.
        range_score = _clamp(
            100.0
            - max(0.0, mean_offset - 0.12) * 85.0
            - max(0.0, max_offset - 0.30) * 20.0
        )

    mean_offset = sum(offsets) / max(len(offsets), 1) if offsets else None
    max_offset = max(offsets) if offsets else None
    bounds_score = 100.0 * (1.0 - out_of_bounds / max(len(events), 1))
    sync_score = _clamp(range_score * 0.72 + text_match * 0.20 + bounds_score * 0.08)
    return {
        "subtitle_cue_count": len(events),
        "subtitle_text_match_percent": round(text_match, 2),
        "subtitle_sync_score": round(sync_score, 2),
        "subtitle_mean_offset_seconds": round(mean_offset, 3) if mean_offset is not None else None,
        "subtitle_max_offset_seconds": round(max_offset, 3) if max_offset is not None else None,
        "subtitle_out_of_bounds_cues": out_of_bounds,
        "subtitle_timeline_coverage_percent": round(timeline_coverage, 2),
        "subtitle_sync_source": sync_source,
    }


def _word_timing_offsets(
    events: list[SubtitleEvent],
    word_timings: list[dict],
    media_duration: float,
) -> list[float] | None:
    cue_keys = [_alignment_key(item.text) for item in events]
    timed = [(item, _alignment_key(str(item.get("text") or ""))) for item in word_timings]
    timed = [(item, key) for item, key in timed if key]
    if (
        not cue_keys
        or not timed
        or any(not key for key in cue_keys)
        or "".join(cue_keys) != "".join(key for _, key in timed)
    ):
        return None

    word_spans: list[tuple[int, int, dict]] = []
    cursor = 0
    for item, key in timed:
        word_spans.append((cursor, cursor + len(key), item))
        cursor += len(key)

    offsets: list[float] = []
    cursor = 0
    for event, key in zip(events, cue_keys):
        start_index, end_index = cursor, cursor + len(key)
        covered = [
            item
            for word_start, word_end, item in word_spans
            if word_end > start_index and word_start < end_index
        ]
        if not covered:
            return None
        expected_start = max(0.0, float(covered[0]["start"]) - 0.08)
        expected_end = float(covered[-1]["end"]) + 0.18
        if media_duration > 0:
            expected_end = min(media_duration, expected_end)
        offsets.extend((abs(event.start - expected_start), abs(event.end - expected_end)))
        cursor = end_index
    return offsets


def _voice_range_offsets(
    events: list[SubtitleEvent],
    package: VerifiedReviewPackage,
) -> tuple[list[float], float]:
    decisions = package.edit_decision_list
    offsets: list[float] = []
    overlap_scores: list[float] = []
    for event in events:
        event_tokens = _tokens(event.text)
        best = None
        best_text_score = -1.0
        for decision in decisions:
            decision_tokens = _tokens(decision.narration)
            text_score = len(event_tokens & decision_tokens) / max(len(event_tokens), 1)
            if text_score > best_text_score:
                best = decision
                best_text_score = text_score
        if best is None:
            offsets.append(max(0.0, event.end - event.start))
            overlap_scores.append(0.0)
            continue
        overlap = max(0.0, min(event.end, best.voice_end) - max(event.start, best.voice_start))
        event_duration = max(0.01, event.end - event.start)
        overlap_scores.append(min(1.0, overlap / event_duration) * max(0.0, best_text_score))
        if event.end < best.voice_start:
            offsets.append(best.voice_start - event.end)
        elif event.start > best.voice_end:
            offsets.append(event.start - best.voice_end)
        else:
            offsets.append(0.0)
    return offsets, 100.0 * sum(overlap_scores) / max(len(overlap_scores), 1)


def _baseline_scores(package: VerifiedReviewPackage, signals: dict) -> dict[str, float]:
    report = package.quality_report
    content = (
        report.evidence_score * 0.38
        + report.direct_visual_match_percent * 0.34
        + report.character_consistency_score * 0.16
        + report.source_coverage_score * 0.12
    )
    translation = (
        report.evidence_score * 0.42
        + report.character_consistency_score * 0.18
        + report.story_coherence_score * 0.15
        + float(signals.get("subtitle_text_match_percent") or 0.0) * 0.25
    )
    drift = signals.get("voice_video_drift_seconds")
    voice_video_score = (
        50.0
        if drift is None or signals.get("signal_collection_failed")
        else _clamp(100.0 - max(0.0, float(drift) - 0.10) * 40.0)
    )
    sync = (
        float(signals.get("subtitle_sync_score") or 0.0) * 0.72
        + voice_video_score * 0.20
        + report.chronology_score * 0.08
    )
    coherence = (
        report.story_coherence_score * 0.62
        + report.style_adherence_score * 0.20
        + report.chronology_score * 0.18
    )
    technical = (
        float(signals.get("technical_duration_score") or 0.0) * 0.42
        + float(signals.get("technical_health_score") or 0.0) * 0.28
        + report.duration_adherence_score * 0.20
        + report.direct_visual_match_percent * 0.10
    )
    return {
        "content_fidelity": _clamp(content),
        "translation_accuracy": _clamp(translation),
        "av_subtitle_sync": _clamp(sync),
        "narrative_coherence": _clamp(coherence),
        "technical_quality": _clamp(technical),
        "safety_compliance": 100.0,
    }


async def _ask_final_judge(
    rendered_video: Path,
    package: VerifiedReviewPackage,
    work_dir: Path,
    signals: dict,
    subtitle_file: Path | None,
    source_transcript_file: Path | None,
) -> dict:
    subtitle_events = _read_subtitles(subtitle_file)
    source_events = _read_subtitles(source_transcript_file)
    scene_by_id = {item.scene_id: item for item in package.scenes}
    event_by_id = {item.event_id: item for item in package.events}
    segment_by_id = {item.segment_id: item for item in package.narration_segments}
    segment_contracts: list[dict] = []
    for decision in _uniform_sample(package.edit_decision_list, 100):
        selected = decision.source_clips[0] if decision.source_clips else None
        segment = segment_by_id.get(decision.segment_id)
        event = event_by_id.get(decision.event_id)
        scene = scene_by_id.get(selected.scene_id) if selected else None
        segment_contracts.append(
            {
                "segment_id": decision.segment_id,
                "narration": decision.narration,
                "verified_event": event.summary if event else "",
                "event_cause": event.cause if event else "",
                "event_consequence": event.consequence if event else "",
                "source_dialogue": scene.evidence.dialogue if scene else [],
                "source_ocr_subtitle": scene.evidence.subtitle if scene else [],
                "visible_actions": scene.visible_actions if scene else [],
                "required_visuals": segment.required_visuals.model_dump() if segment else {},
                "selected_scene_match_score": round(selected.match_score * 100.0, 2) if selected else 0.0,
                "voice_start": decision.voice_start,
                "voice_end": decision.voice_end,
            }
        )

    evidence_payload = {
        "video": {
            "path_name": rendered_video.name,
            "title": package.title,
            "target_minutes": package.target_minutes,
        },
        "deterministic_signals": signals,
        "post_render_quality": package.quality_report.model_dump(mode="json"),
        "verified_events": [
            {
                "event_id": item.event_id,
                "order_index": item.order_index,
                "summary": item.summary,
                "cause": item.cause,
                "consequence": item.consequence,
                "confidence": item.confidence,
            }
            for item in _uniform_sample(package.events, 100)
        ],
        "review_segments_with_source_evidence": segment_contracts,
        "rendered_subtitles": [
            {"start": item.start, "end": item.end, "text": item.text}
            for item in _uniform_sample(subtitle_events, 100)
        ],
        "source_transcript_sample": [
            {"start": item.start, "end": item.end, "text": item.text}
            for item in _uniform_sample(source_events, 100)
        ],
    }
    content: list[dict] = [
        {
            "type": "text",
            "text": json.dumps(evidence_payload, ensure_ascii=False),
        }
    ]
    for frame in _select_final_frames(work_dir, 9):
        content.append({"type": "text", "text": f"FRAME MẪU: {frame.name}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(frame.read_bytes()).decode('ascii')}"},
            }
        )

    system_prompt = (
        "Bạn là giám khảo độc lập chấm BẢN VIDEO REVIEW ĐÃ RENDER, không phải người viết kịch bản. "
        "Chỉ kết luận từ verified events/evidence, source transcript, QA frames và số đo deterministic. "
        "translation_accuracy chấm độ trung thành ngữ nghĩa của lời Việt/phụ đề với evidence nguồn, tên riêng, "
        "quan hệ và kết quả; đây là bản review cô đọng nên không đòi dịch từng câu theo nghĩa đen. "
        "av_subtitle_sync phải tôn trọng subtitle_sync_score, offset, cue bounds và voice_video_drift; không đoán "
        "timing chỉ từ một frame tĩnh. technical_quality xét frame đen/trống, thời lượng, khả năng đọc subtitle, "
        "hình và audio timeline. safety_compliance không phạt cảnh bạo lực/hư cấu chỉ vì nó thuộc nội dung phim; "
        "chỉ phạt cách review cổ súy hành vi nguy hiểm, thù ghét, tình dục hóa trẻ vị thành niên, lộ dữ liệu cá nhân, "
        "bịa đặt gây hại hoặc mô tả phản cảm không cần thiết. Không nâng điểm để lịch sự. "
        "Trả duy nhất JSON: {summary:string,strengths:string[],recommendations:string[],criteria:{"
        "content_fidelity:{score:0..100,feedback:string,findings:string[]},"
        "translation_accuracy:{score:0..100,feedback:string,findings:string[]},"
        "av_subtitle_sync:{score:0..100,feedback:string,findings:string[]},"
        "narrative_coherence:{score:0..100,feedback:string,findings:string[]},"
        "technical_quality:{score:0..100,feedback:string,findings:string[]},"
        "safety_compliance:{score:0..100,feedback:string,findings:string[]}}}."
    )
    response = await _client().chat.completions.create(
        model=settings.review_ai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        timeout=240,
    )
    return _loads_json(response.choices[0].message.content or "{}")


def _build_evaluation(
    payload: dict,
    baselines: dict[str, float],
    signals: dict,
    *,
    fallback_used: bool,
) -> FinalReviewEvaluation:
    raw_criteria = payload.get("criteria")
    criteria_by_key: dict[str, dict] = {}
    if isinstance(raw_criteria, dict):
        criteria_by_key = {
            str(key): value
            for key, value in raw_criteria.items()
            if str(key) in _CRITERION_KEYS and isinstance(value, dict)
        }
    elif isinstance(raw_criteria, list):
        criteria_by_key = {
            str(item.get("key")): item
            for item in raw_criteria
            if isinstance(item, dict) and str(item.get("key")) in _CRITERION_KEYS
        }

    criteria: list[FinalReviewCriterion] = []
    for key, label, weight in _CRITERIA:
        baseline = _clamp(float(baselines.get(key, 0.0)))
        raw = criteria_by_key.get(key)
        ai_score = _optional_score(raw.get("score")) if raw else None
        if ai_score is None:
            # Evidence coverage is not a semantic source/translation
            # comparison. Keep translation conservative when the judge did
            # not return a valid score.
            score = min(baseline, 60.0) if key == "translation_accuracy" else baseline
            feedback = _fallback_feedback(key, score, signals)
            findings = _deterministic_findings(key, signals)
            fallback_used = True
        else:
            influence = _AI_INFLUENCE[key]
            score = ai_score if key == "safety_compliance" else baseline * (1.0 - influence) + ai_score * influence
            feedback = _safe_text(raw.get("feedback"), _fallback_feedback(key, score, signals))
            findings = _safe_string_list(raw.get("findings"), 8)
            for finding in _deterministic_findings(key, signals):
                if finding not in findings:
                    findings.append(finding)
        score = round(_clamp(score), 2)
        criteria.append(
            FinalReviewCriterion(
                key=key,
                label=label,
                score=score,
                weight_percent=weight,
                passed=score >= _CRITERION_PASS_SCORE[key],
                feedback=feedback,
                findings=findings[:10],
            )
        )

    overall = round(
        sum(item.score * item.weight_percent for item in criteria)
        / max(sum(item.weight_percent for item in criteria), 1.0),
        2,
    )
    passed = (
        overall >= 80.0
        and bool(signals.get("post_render_passed"))
        and all(item.passed for item in criteria)
    )
    verdict = (
        "excellent"
        if overall >= 90.0 and passed
        else "good"
        if overall >= 80.0 and passed
        else "needs_improvement"
        if overall >= 65.0
        else "poor"
    )
    strengths = _safe_string_list(payload.get("strengths"), 8)
    if not strengths:
        strengths = [
            f"{item.label}: {item.score:.1f}/100"
            for item in sorted(criteria, key=lambda value: value.score, reverse=True)[:2]
        ]
    recommendations = _safe_string_list(payload.get("recommendations"), 10)
    if not recommendations:
        recommendations = [
            item.feedback
            for item in sorted(criteria, key=lambda value: value.score)
            if item.score < 85.0 and item.feedback
        ][:4]
    if not signals.get("post_render_passed"):
        qa_recommendation = "Bản xem trước chưa qua post-render QA; cần sửa hết lỗi QA trước khi xuất bản final."
        if qa_recommendation not in recommendations:
            recommendations.insert(0, qa_recommendation)
    summary = _safe_text(
        payload.get("summary"),
        (
            f"Video đạt {overall:.1f}/100 qua 6 tiêu chí. "
            + ("Bản render đạt chuẩn đánh giá cuối." if passed else "Bản render còn tiêu chí cần cải thiện.")
        ),
    )
    if not signals.get("post_render_passed"):
        summary = f"Bản xem trước chưa đạt post-render QA. {summary}"
    return FinalReviewEvaluation(
        overall_score=overall,
        verdict=verdict,
        passed=passed,
        summary=summary,
        strengths=strengths,
        recommendations=recommendations,
        criteria=criteria,
        model=settings.review_ai_model,
        fallback_used=fallback_used,
        evaluated_at=datetime.now(UTC).isoformat(),
    )


def _deterministic_findings(key: str, signals: dict) -> list[str]:
    if key == "av_subtitle_sync":
        offset = signals.get("subtitle_mean_offset_seconds")
        offset_text = "không đo được" if offset is None else f"{float(offset):.3f}s"
        drift = signals.get("voice_video_drift_seconds")
        drift_text = "không đo được" if drift is None else f"{float(drift):.3f}s"
        return [
            f"Điểm timing phụ đề đo được: {float(signals.get('subtitle_sync_score') or 0.0):.1f}/100.",
            f"Lệch trung bình phụ đề/word timing: {offset_text}; lệch video/voice: {drift_text}.",
        ]
    if key == "technical_quality":
        if signals.get("signal_collection_failed"):
            return ["Không thu đủ số đo media cuối; điểm kỹ thuật đang dùng fallback thận trọng."]
        return [
            f"Video dài {float(signals.get('media_duration_seconds') or 0.0):.2f}s; file {int(signals.get('video_file_size_bytes') or 0)} byte.",
        ]
    if key == "translation_accuracy":
        return [
            f"Phụ đề giữ {float(signals.get('subtitle_text_match_percent') or 0.0):.1f}% nội dung lời review đã đọc.",
        ]
    return []


def _fallback_feedback(key: str, score: float, signals: dict) -> str:
    label = next(label for item_key, label, _ in _CRITERIA if item_key == key)
    if key == "translation_accuracy":
        return (
            "Chưa kiểm chứng trực tiếp chuyển ngữ với transcript nguồn vì AI giám khảo không trả kết quả; "
            f"điểm fallback được giới hạn thận trọng ở {score:.1f}/100."
        )
    if key == "av_subtitle_sync" and not signals.get("subtitle_cue_count"):
        return "Không tìm thấy cue phụ đề để xác nhận đồng bộ với giọng đọc."
    level = "đạt" if score >= _CRITERION_PASS_SCORE[key] else "chưa đạt"
    return f"{label} {level} theo số đo hậu kỳ ({score:.1f}/100)."


def _select_final_frames(work_dir: Path, limit: int) -> list[Path]:
    qa_dir = work_dir / "review_qa_frames"
    paths = sorted(
        path
        for path in qa_dir.glob("*.jpg")
        if path.is_file() and path.stat().st_size > 0
    )
    return _uniform_sample(paths, limit)


def _uniform_sample(values: list, limit: int) -> list:
    if len(values) <= limit:
        return list(values)
    if limit <= 1:
        return [values[len(values) // 2]]
    indices = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[index] for index in indices]


def _alignment_key(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text or "")).casefold()
    return "".join(char for char in normalized if char.isalnum())


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", str(text or "")).casefold()
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return {item for item in re.findall(r"[\w]+", normalized, flags=re.UNICODE) if item}


def _loads_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    return value if isinstance(value, dict) else {}


def _optional_score(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return _clamp(number)


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _safe_text(value: object, fallback: str) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).strip()
    return text or fallback


def _safe_string_list(value: object, limit: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = unicodedata.normalize("NFC", str(item or "")).strip()
        if text and text not in result:
            result.append(text)
    return result[:limit]


def _client() -> AsyncOpenAI:
    # Shared pool: a per-call client leaked an httpx connection pool per request.
    # timeout/max_retries preserve the previous SDK behaviour for this module.
    return get_async_openai(
        settings.ninerouter_api_key or "local-ninerouter",
        settings.ninerouter_api_url,
        timeout=240.0,
        max_retries=2,
    )
