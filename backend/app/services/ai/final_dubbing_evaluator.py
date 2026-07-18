from __future__ import annotations

import asyncio
import json
import math
import re
import subprocess
import unicodedata
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from openai import AsyncOpenAI

from app.core import settings
from app.models.review import FinalReviewCriterion, FinalReviewCriterionKey, FinalReviewEvaluation
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
_PASS_SCORE = {
    "content_fidelity": 75.0,
    "translation_accuracy": 70.0,
    "av_subtitle_sync": 75.0,
    "narrative_coherence": 70.0,
    "technical_quality": 70.0,
    "safety_compliance": 80.0,
}
_AI_INFLUENCE = {
    "content_fidelity": 0.75,
    "translation_accuracy": 0.85,
    # Measured timeline/media facts remain authoritative for these criteria.
    "av_subtitle_sync": 0.15,
    "narrative_coherence": 0.65,
    "technical_quality": 0.20,
    "safety_compliance": 1.0,
}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


@dataclass(frozen=True)
class DubbingEvaluationArtifacts:
    subtitle_file: Path | None = None
    source_subtitle_file: Path | None = None
    narration_audio: Path | None = None
    source_video: Path | None = None


def discover_dubbing_evaluation_artifacts(work_dir: Path) -> DubbingEvaluationArtifacts:
    """Find artifacts retained by both current and older dubbing jobs."""

    target_candidates = [
        work_dir / f"subtitles.{settings.target_language}.srt",
        work_dir / "subtitles.grouped.srt",
        work_dir / "subtitles.whisper.srt",
    ]
    subtitle_file = _first_file(target_candidates)
    source_candidates = [
        work_dir / "subtitles.whisper.srt",
        work_dir / "subtitles.grouped.srt",
        *sorted(work_dir.glob("subtitles*.srt")),
    ]
    source_subtitle_file = _first_file(
        path
        for path in source_candidates
        if path != subtitle_file and "positioned" not in path.name
    )
    if source_subtitle_file is None and subtitle_file is not None:
        # Passthrough/same-language jobs intentionally have one transcript.
        source_subtitle_file = subtitle_file

    source_video = _first_file(
        path
        for path in sorted(work_dir.glob("source*"))
        if path.suffix.lower() in _VIDEO_SUFFIXES
    )
    return DubbingEvaluationArtifacts(
        subtitle_file=subtitle_file,
        source_subtitle_file=source_subtitle_file,
        narration_audio=_first_file([work_dir / "narration_timed.wav"]),
        source_video=source_video,
    )


async def evaluate_final_dubbing_video(
    rendered_video: Path,
    work_dir: Path,
    *,
    subtitle_file: Path | None = None,
    source_subtitle_file: Path | None = None,
    narration_audio: Path | None = None,
    source_video: Path | None = None,
    subtitles_expected: bool = True,
    narration_expected: bool = True,
    target_subtitle_burned: bool = True,
    translation_expected: bool = True,
    source_language: str = "auto",
    target_language: str | None = None,
    video_speed: float = 1.0,
) -> FinalReviewEvaluation:
    """Score an ordinary translated/dubbed output without gating its completion.

    Semantic criteria use an AI judge when source and target transcripts exist.
    Duration, cue bounds and the per-cue TTS timeline contract are measured
    locally. Missing artifacts or AI/network failures always produce a complete,
    explicit fallback scorecard and never invalidate an already-rendered MP4.
    """

    work_dir.mkdir(parents=True, exist_ok=True)
    target_language = target_language or settings.target_language
    try:
        signals = await _collect_dubbing_signals(
            rendered_video,
            work_dir,
            subtitle_file,
            source_subtitle_file,
            narration_audio,
            source_video,
            subtitles_expected=subtitles_expected,
            narration_expected=narration_expected,
            target_subtitle_burned=target_subtitle_burned,
            translation_expected=translation_expected,
            source_language=source_language,
            target_language=target_language,
            video_speed=video_speed,
        )
    except Exception as exc:
        signals = _unavailable_signals(
            rendered_video,
            subtitles_expected=subtitles_expected,
            narration_expected=narration_expected,
            reason=f"Không thu được số đo hậu kỳ: {exc}",
        )

    baselines = _baseline_scores(signals)
    payload: dict = {}
    fallback_used = bool(signals.get("unavailable_reasons"))
    try:
        if signals.get("semantic_comparison_available"):
            payload = await _ask_dubbing_judge(
                rendered_video,
                signals,
                _read_subtitles(source_subtitle_file),
                _read_subtitles(subtitle_file),
                source_language=source_language,
                target_language=target_language,
            )
        else:
            fallback_used = True
    except Exception as exc:
        print(f"Final dubbing evaluator fallback: {exc}")
        fallback_used = True
        payload = {}

    try:
        evaluation = _build_dubbing_evaluation(
            payload,
            baselines,
            signals,
            fallback_used=fallback_used,
        )
    except Exception as exc:
        evaluation = emergency_dubbing_evaluation(f"Không dựng được bảng điểm: {exc}")

    try:
        (work_dir / "final_evaluation.json").write_text(
            evaluation.model_dump_json(indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"Cannot persist final dubbing evaluation: {exc}")
    return evaluation


def emergency_dubbing_evaluation(reason: str) -> FinalReviewEvaluation:
    reason = _safe_text(reason, "Không đủ dữ liệu để đánh giá video đầu ra.")
    return FinalReviewEvaluation(
        overall_score=0.0,
        verdict="poor",
        passed=False,
        summary=f"Chưa thể đánh giá tin cậy video đã render. {reason}",
        recommendations=[reason, "Giữ video đầu ra và thử chấm lại khi artifact/AI khả dụng."],
        criteria=[
            FinalReviewCriterion(
                key=key,
                label=label,
                score=0.0,
                weight_percent=weight,
                passed=False,
                feedback=reason,
                findings=[reason],
            )
            for key, label, weight in _CRITERIA
        ],
        model=settings.review_ai_model,
        fallback_used=True,
        evaluated_at=datetime.now(UTC).isoformat(),
    )


def dubbing_evaluation_stage(evaluation: FinalReviewEvaluation) -> str:
    judge = "Chấm dự phòng" if evaluation.fallback_used else "AI chấm"
    return f"Hoàn tất pipeline · {judge} {evaluation.overall_score:.1f}/100"


async def _collect_dubbing_signals(
    rendered_video: Path,
    work_dir: Path,
    subtitle_file: Path | None,
    source_subtitle_file: Path | None,
    narration_audio: Path | None,
    source_video: Path | None,
    *,
    subtitles_expected: bool,
    narration_expected: bool,
    target_subtitle_burned: bool,
    translation_expected: bool,
    source_language: str,
    target_language: str,
    video_speed: float,
) -> dict:
    ffmpeg = find_ffmpeg() or "ffmpeg"
    speed = max(0.01, float(video_speed or 1.0))
    output_duration = await _probe_duration(ffmpeg, rendered_video)
    source_duration = await _probe_duration(ffmpeg, source_video)
    narration_duration = await _probe_duration(ffmpeg, narration_audio)
    output_has_video, output_has_audio = await _probe_streams(ffmpeg, rendered_video)
    target_events = _read_subtitles(subtitle_file)
    source_events = _read_subtitles(source_subtitle_file)
    (
        tts_segment_count,
        tts_expected_count,
        tts_cue_coverage,
        tts_manifest_matched,
        tts_overrun_count,
    ) = _tts_segment_metrics(
        work_dir,
        target_events,
        narration_duration,
    )
    rendered_events = [
        SubtitleEvent(item.start / speed, item.end / speed, item.text)
        for item in target_events
    ]

    unavailable: list[str] = []
    output_ok = (
        rendered_video.is_file()
        and rendered_video.stat().st_size > 0
        and output_duration > 0
        and output_has_video is not False
    )
    if not output_ok:
        unavailable.append("File video đầu ra bị thiếu, rỗng hoặc không đọc được thời lượng.")
    if subtitles_expected and not target_events:
        unavailable.append("Thiếu phụ đề đích nên chưa xác nhận được nội dung và timecode.")
    if narration_expected and not (narration_audio and narration_audio.is_file() and narration_duration > 0):
        unavailable.append("Thiếu narration_timed.wav; có thể video đã fallback về âm thanh gốc.")
    if narration_expected and target_events and not tts_manifest_matched:
        unavailable.append(
            "Artifact TTS per-cue không khớp đầy đủ với cue phụ đề; sync chỉ được chấm ở mức thận trọng."
        )
    if narration_expected and output_has_audio is False:
        unavailable.append("Video đầu ra không có audio stream để xác nhận phần lồng tiếng.")
    if subtitles_expected and not source_events:
        unavailable.append("Thiếu transcript nguồn nên AI không thể đối chiếu trực tiếp bản dịch.")

    out_of_bounds = sum(
        item.start < -0.01
        or item.end <= item.start
        or (output_duration > 0 and item.end > output_duration + 0.20)
        for item in rendered_events
    )
    overlaps = sum(
        current.start < previous.end - 0.02
        for previous, current in zip(rendered_events, rendered_events[1:])
    )
    empty_cues = sum(not _alignment_key(item.text) for item in target_events)
    visible_duration = sum(max(0.0, item.end - item.start) for item in rendered_events)
    coverage = 100.0 * visible_duration / max(output_duration, 0.01) if rendered_events else 0.0
    reading_rates = [
        len(_alignment_key(item.text)) / max(item.end - item.start, 0.05)
        for item in rendered_events
        if _alignment_key(item.text)
    ]
    mean_reading_rate = sum(reading_rates) / max(len(reading_rates), 1)
    max_reading_rate = max(reading_rates, default=0.0)
    readability_score = _clamp(
        100.0
        - max(0.0, mean_reading_rate - 17.0) * 4.0
        - max(0.0, max_reading_rate - 25.0) * 1.5
    )

    cue_alignment_score, mean_source_target_offset = _source_target_timeline_score(
        source_events,
        target_events,
    )
    bounds_score = 100.0 if not rendered_events else _clamp(
        100.0 - 100.0 * (out_of_bounds + overlaps) / max(len(rendered_events), 1)
    )
    rendered_narration_duration = narration_duration / speed if narration_duration > 0 else 0.0
    narration_drift = (
        abs(rendered_narration_duration - output_duration)
        if output_duration > 0 and rendered_narration_duration > 0
        else None
    )
    voice_duration_score = (
        _clamp(100.0 - max(0.0, narration_drift - 0.12) * 45.0)
        if narration_drift is not None
        else (35.0 if narration_expected else 100.0)
    )
    if not subtitles_expected:
        sync_score = 100.0 if output_ok else 0.0
        sync_source = "not_applicable"
    elif not rendered_events:
        sync_score = 0.0
        sync_source = "missing_subtitles"
    else:
        # synthesize_srt renders one speech slot per SRT cue and never lets it
        # cross the next cue. Duration health verifies that timeline survived
        # the final FFmpeg speed/mix step.
        segment_score = tts_cue_coverage if narration_expected else 100.0
        sync_score = _clamp(
            bounds_score * 0.45
            + voice_duration_score * 0.25
            + cue_alignment_score * 0.05
            + segment_score * 0.25
        )
        if narration_expected and not tts_manifest_matched:
            sync_score = min(sync_score, 70.0)
        if narration_expected and output_has_audio is False:
            sync_score = min(sync_score, 40.0)
        sync_source = (
            "per_cue_tts_contract"
            if tts_manifest_matched
            else "legacy_or_unverified_tts"
            if narration_duration > 0
            else "subtitle_timeline_only"
        )

    expected_output_duration = source_duration / speed if source_duration > 0 else 0.0
    output_drift = (
        abs(output_duration - expected_output_duration)
        if output_duration > 0 and expected_output_duration > 0
        else None
    )
    duration_score = (
        _clamp(100.0 - max(0.0, output_drift - 0.15) * 35.0)
        if output_drift is not None
        else (70.0 if output_ok else 0.0)
    )
    same_language = source_language.lower() == target_language.lower() and source_language.lower() != "auto"
    same_transcript = (
        bool(source_events and target_events)
        and _alignment_key(" ".join(item.text for item in source_events))
        == _alignment_key(" ".join(item.text for item in target_events))
    )
    translation_required = subtitles_expected and translation_expected and not same_language
    if translation_required and same_transcript:
        unavailable.append(
            "Transcript đích giống hệt transcript nguồn dù job yêu cầu dịch; cần kiểm tra nguy cơ chưa chuyển ngữ."
        )
    semantic_available = bool(source_events and target_events and translation_required)

    return {
        "output_file_size_bytes": rendered_video.stat().st_size if rendered_video.is_file() else 0,
        "output_duration_seconds": round(output_duration, 3),
        "source_duration_seconds": round(source_duration, 3),
        "expected_output_duration_seconds": round(expected_output_duration, 3),
        "output_duration_drift_seconds": round(output_drift, 3) if output_drift is not None else None,
        "narration_duration_seconds": round(narration_duration, 3),
        "rendered_narration_drift_seconds": round(narration_drift, 3) if narration_drift is not None else None,
        "output_media_ok": output_ok,
        "output_has_video_stream": output_has_video,
        "output_has_audio_stream": output_has_audio,
        "subtitles_expected": subtitles_expected,
        "narration_expected": narration_expected,
        "target_subtitle_burned": target_subtitle_burned,
        "target_cue_count": len(target_events),
        "source_cue_count": len(source_events),
        "subtitle_out_of_bounds_cues": out_of_bounds,
        "subtitle_overlap_count": overlaps,
        "subtitle_empty_cues": empty_cues,
        "subtitle_timeline_coverage_percent": round(min(100.0, coverage), 2),
        "subtitle_mean_chars_per_second": round(mean_reading_rate, 2),
        "subtitle_max_chars_per_second": round(max_reading_rate, 2),
        "subtitle_readability_score": round(readability_score, 2),
        "source_target_timeline_score": round(cue_alignment_score, 2),
        "source_target_mean_offset_seconds": (
            round(mean_source_target_offset, 3) if mean_source_target_offset is not None else None
        ),
        "subtitle_sync_score": round(sync_score, 2),
        "subtitle_sync_source": sync_source,
        "tts_segment_count": tts_segment_count,
        "tts_expected_segment_count": tts_expected_count,
        "tts_cue_coverage_percent": round(tts_cue_coverage, 2),
        "tts_manifest_matched": tts_manifest_matched,
        "tts_overrun_segment_count": tts_overrun_count,
        "voice_duration_score": round(voice_duration_score, 2),
        "technical_duration_score": round(duration_score, 2),
        "translation_required": translation_required,
        "source_target_text_identical": same_transcript,
        "semantic_comparison_available": semantic_available,
        "unavailable_reasons": unavailable,
    }


def _unavailable_signals(
    rendered_video: Path,
    *,
    subtitles_expected: bool,
    narration_expected: bool,
    reason: str,
) -> dict:
    return {
        "output_file_size_bytes": rendered_video.stat().st_size if rendered_video.is_file() else 0,
        "output_duration_seconds": 0.0,
        "source_duration_seconds": 0.0,
        "expected_output_duration_seconds": 0.0,
        "output_duration_drift_seconds": None,
        "narration_duration_seconds": 0.0,
        "rendered_narration_drift_seconds": None,
        "output_media_ok": False,
        "output_has_video_stream": None,
        "output_has_audio_stream": None,
        "subtitles_expected": subtitles_expected,
        "narration_expected": narration_expected,
        "target_subtitle_burned": False,
        "target_cue_count": 0,
        "source_cue_count": 0,
        "subtitle_out_of_bounds_cues": 0,
        "subtitle_overlap_count": 0,
        "subtitle_empty_cues": 0,
        "subtitle_timeline_coverage_percent": 0.0,
        "subtitle_mean_chars_per_second": 0.0,
        "subtitle_max_chars_per_second": 0.0,
        "subtitle_readability_score": 0.0,
        "source_target_timeline_score": 0.0,
        "source_target_mean_offset_seconds": None,
        "subtitle_sync_score": 0.0,
        "subtitle_sync_source": "unavailable",
        "tts_segment_count": 0,
        "tts_expected_segment_count": 0,
        "tts_cue_coverage_percent": 0.0,
        "tts_manifest_matched": False,
        "tts_overrun_segment_count": 0,
        "voice_duration_score": 0.0,
        "technical_duration_score": 0.0,
        "translation_required": subtitles_expected,
        "source_target_text_identical": False,
        "semantic_comparison_available": False,
        "unavailable_reasons": [reason],
    }


def _baseline_scores(signals: dict) -> dict[str, float]:
    if not signals.get("subtitles_expected"):
        content = translation = coherence = 100.0
    else:
        timeline = float(signals.get("source_target_timeline_score") or 0.0)
        target_count = int(signals.get("target_cue_count") or 0)
        source_count = int(signals.get("source_cue_count") or 0)
        # Translation may split one source cue into several shorter target
        # cues. Timeline coverage is authoritative; raw cue-count equality is
        # not a quality requirement.
        cue_coverage = timeline if source_count else 45.0
        empty_score = 100.0 * (1.0 - int(signals.get("subtitle_empty_cues") or 0) / max(target_count, 1))
        content = timeline * 0.45 + cue_coverage * 0.35 + empty_score * 0.20
        translation = 100.0 if not signals.get("translation_required") else min(content, 60.0)
        coherence = empty_score * 0.45 + timeline * 0.30 + float(signals.get("subtitle_readability_score") or 0.0) * 0.25

    stream_score = 70.0
    if signals.get("output_has_video_stream") is not None:
        video_score = 100.0 if signals.get("output_has_video_stream") else 0.0
        audio_score = 100.0 if signals.get("output_has_audio_stream") else 0.0
        stream_score = video_score * 0.55 + audio_score * 0.45
    technical = (
        float(signals.get("technical_duration_score") or 0.0) * 0.55
        + (100.0 if signals.get("output_media_ok") else 0.0) * 0.25
        + stream_score * 0.20
    )
    if signals.get("narration_expected") and signals.get("output_has_audio_stream") is False:
        technical = min(technical, 50.0)
    return {
        "content_fidelity": _clamp(content),
        "translation_accuracy": _clamp(translation),
        "av_subtitle_sync": _clamp(float(signals.get("subtitle_sync_score") or 0.0)),
        "narrative_coherence": _clamp(coherence),
        "technical_quality": _clamp(technical),
        "safety_compliance": 100.0,
    }


async def _ask_dubbing_judge(
    rendered_video: Path,
    signals: dict,
    source_events: list[SubtitleEvent],
    target_events: list[SubtitleEvent],
    *,
    source_language: str,
    target_language: str,
) -> dict:
    aligned_pairs = _aligned_semantic_pairs(source_events, target_events)
    pairs = [aligned_pairs[index] for index in _sample_indices(len(aligned_pairs), 120)]
    evidence = {
        "rendered_video_name": rendered_video.name,
        "source_language": source_language,
        "target_language": target_language,
        "deterministic_signals": signals,
        "aligned_subtitle_samples": pairs,
    }
    system_prompt = (
        "Bạn là giám khảo độc lập chấm BẢN VIDEO DỊCH/LỒNG TIẾNG ĐÃ RENDER. "
        "Chỉ kết luận từ transcript nguồn, phụ đề đích và số đo deterministic được cung cấp. "
        "Bạn KHÔNG được cung cấp frame hình, waveform hay audio để nghe; không được tuyên bố đã xem/nghe video, "
        "không được kết luận lip-sync hoặc đồng bộ từng từ. "
        "content_fidelity: có bỏ sót, thêm sai, đảo nghĩa, sai chủ thể/sự kiện hay không. "
        "translation_accuracy: đúng nghĩa, tự nhiên, nhất quán tên riêng, vai vế và cách xưng hô. "
        "narrative_coherence: câu nối tiếp dễ hiểu, không đứt mạch. "
        "av_subtitle_sync và technical_quality phải tôn trọng số đo duration, cue bounds và sync; "
        "không được nâng điểm trái số đo. safety_compliance đánh giá cách chuyển ngữ có cổ súy nguy hiểm, "
        "thù ghét, tình dục hóa trẻ vị thành niên, lộ dữ liệu cá nhân hoặc thêm nội dung gây hại hay không. "
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
            {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        timeout=240,
    )
    return _loads_json(response.choices[0].message.content or "{}")


def _build_dubbing_evaluation(
    payload: dict,
    baselines: dict[str, float],
    signals: dict,
    *,
    fallback_used: bool,
) -> FinalReviewEvaluation:
    raw_criteria = payload.get("criteria")
    criteria_by_key = raw_criteria if isinstance(raw_criteria, dict) else {}
    criteria: list[FinalReviewCriterion] = []
    for key, label, weight in _CRITERIA:
        baseline = _clamp(float(baselines.get(key, 0.0)))
        raw = criteria_by_key.get(key) if isinstance(criteria_by_key.get(key), dict) else None
        ai_score = _optional_score(raw.get("score")) if raw else None
        if ai_score is None:
            score = baseline
            if signals.get("translation_required"):
                fallback_caps = {
                    "content_fidelity": 65.0,
                    "translation_accuracy": (
                        35.0 if signals.get("source_target_text_identical") else 60.0
                    ),
                    "narrative_coherence": 70.0,
                    "safety_compliance": 75.0,
                }
                score = min(score, fallback_caps.get(key, score))
            feedback = _fallback_feedback(key, score, signals)
            findings = _deterministic_findings(key, signals)
            fallback_used = True
        else:
            influence = _AI_INFLUENCE[key]
            score = ai_score if key == "safety_compliance" else baseline * (1.0 - influence) + ai_score * influence
            if key in {"av_subtitle_sync", "technical_quality"}:
                # The general dubbing pipeline currently retains cue-slot and
                # duration evidence, not word-level boundaries or lip-sync
                # measurements. Never let model prose overclaim what was seen.
                if key == "technical_quality":
                    score = baseline
                feedback = _fallback_feedback(key, score, signals)
                findings = _deterministic_findings(key, signals)
            else:
                feedback = _safe_text(raw.get("feedback"), _fallback_feedback(key, score, signals))
                findings = _safe_string_list(raw.get("findings"), 8)
            for finding in _deterministic_findings(key, signals):
                if finding not in findings:
                    findings.append(finding)
        if key == "av_subtitle_sync" and signals.get("narration_expected"):
            if not signals.get("tts_manifest_matched"):
                score = min(score, 70.0)
            if signals.get("output_has_audio_stream") is False:
                score = min(score, 40.0)
            feedback = _fallback_feedback(key, score, signals)
            findings = _deterministic_findings(key, signals)
        if (
            key == "technical_quality"
            and signals.get("narration_expected")
            and signals.get("output_has_audio_stream") is False
        ):
            score = min(score, 50.0)
            feedback = _fallback_feedback(key, score, signals)
            findings = _deterministic_findings(key, signals)
        score = round(_clamp(score), 2)
        criteria.append(
            FinalReviewCriterion(
                key=key,
                label=label,
                score=score,
                weight_percent=weight,
                passed=score >= _PASS_SCORE[key],
                feedback=feedback,
                findings=findings[:10],
            )
        )

    overall = round(sum(item.score * item.weight_percent for item in criteria) / 100.0, 2)
    critical_ok = bool(signals.get("output_media_ok"))
    if signals.get("subtitles_expected"):
        critical_ok = critical_ok and bool(signals.get("target_cue_count"))
    if signals.get("narration_expected"):
        critical_ok = (
            critical_ok
            and float(signals.get("narration_duration_seconds") or 0.0) > 0
            and signals.get("output_has_audio_stream") is True
        )
    critical_ok = critical_ok and signals.get("output_has_video_stream") is True
    passed = overall >= 80.0 and critical_ok and all(item.passed for item in criteria)
    verdict = (
        "excellent" if overall >= 90.0 and passed else
        "good" if overall >= 80.0 and passed else
        "needs_improvement" if overall >= 65.0 else
        "poor"
    )
    strengths = _safe_string_list(payload.get("strengths"), 8) or [
        f"{item.label}: {item.score:.1f}/100"
        for item in sorted(criteria, key=lambda value: value.score, reverse=True)[:2]
    ]
    recommendations = _safe_string_list(payload.get("recommendations"), 10)
    for reason in signals.get("unavailable_reasons") or []:
        if reason not in recommendations:
            recommendations.insert(0, reason)
    if not recommendations:
        recommendations = [
            item.feedback
            for item in sorted(criteria, key=lambda value: value.score)
            if item.score < 85.0 and item.feedback
        ][:4]
    summary = _safe_text(
        payload.get("summary"),
        f"Video đạt {overall:.1f}/100 qua 6 tiêu chí. "
        + ("Bản render đạt chuẩn đánh giá cuối." if passed else "Bản render còn tiêu chí cần cải thiện."),
    )
    if fallback_used:
        summary = f"Bảng điểm có dùng fallback do AI không khả dụng hoặc thiếu một phần dữ liệu. {summary}"
    summary = (
        f"{summary} Phạm vi đồng bộ chỉ xác nhận cue/timecode, duration và artifact TTS; "
        "chưa phải phép đo lip-sync hoặc lệch từng từ."
    )
    return FinalReviewEvaluation(
        overall_score=overall,
        verdict=verdict,
        passed=passed,
        summary=summary,
        strengths=strengths,
        recommendations=recommendations[:10],
        criteria=criteria,
        model=settings.review_ai_model,
        fallback_used=fallback_used,
        evaluated_at=datetime.now(UTC).isoformat(),
    )


def _deterministic_findings(key: str, signals: dict) -> list[str]:
    if key == "av_subtitle_sync":
        drift = signals.get("rendered_narration_drift_seconds")
        drift_text = "không đo được" if drift is None else f"{float(drift):.3f}s"
        if signals.get("subtitle_sync_source") == "not_applicable":
            return ["Job không yêu cầu phụ đề/giọng đọc mới; tiêu chí đồng bộ không áp dụng."]
        per_cue_finding = (
            "Đã đối chiếu đầy đủ artifact speech theo từng cue; mỗi TTS cue được dựng trong slot timecode. "
            "Chưa đo lip-sync hoặc lệch từng từ."
            if signals.get("tts_manifest_matched")
            else "Artifact speech per-cue thiếu hoặc thuộc pipeline cũ; chưa xác minh được từng câu đọc có khớp cue. "
            "Chỉ dùng cue bounds và duration tổng ở mức thận trọng."
        )
        findings = [
            f"Điểm sync deterministic: {float(signals.get('subtitle_sync_score') or 0.0):.1f}/100.",
            f"Lệch thời lượng narration/video: {drift_text}; cue vượt biên: {int(signals.get('subtitle_out_of_bounds_cues') or 0)}.",
            per_cue_finding,
            f"Artifact TTS khớp {int(signals.get('tts_segment_count') or 0)}/{int(signals.get('tts_expected_segment_count') or 0)} cue ({float(signals.get('tts_cue_coverage_percent') or 0.0):.1f}%).",
            f"Số speech WAV vượt slot: {int(signals.get('tts_overrun_segment_count') or 0)}.",
        ]
        if not signals.get("target_subtitle_burned"):
            findings.append(
                "Video giữ phụ đề cứng có sẵn từ nguồn; SRT đích chỉ dùng làm timeline giọng đọc, không được burn lên hình."
            )
        return findings
    if key == "translation_accuracy":
        if not signals.get("translation_required"):
            return ["Job cùng ngôn ngữ/passthrough nên không phát sinh phép dịch cần chấm."]
        return [
            f"Đối chiếu {int(signals.get('source_cue_count') or 0)} cue nguồn với {int(signals.get('target_cue_count') or 0)} cue đích."
        ]
    if key == "technical_quality":
        stream_text = (
            "không đo được"
            if signals.get("output_has_video_stream") is None
            else f"video={'có' if signals.get('output_has_video_stream') else 'thiếu'}, "
            f"audio={'có' if signals.get('output_has_audio_stream') else 'thiếu'}"
        )
        return [
            f"Video dài {float(signals.get('output_duration_seconds') or 0.0):.2f}s; file {int(signals.get('output_file_size_bytes') or 0)} byte.",
            f"Media streams: {stream_text}.",
            "Đã kiểm tra container, duration và sự tồn tại stream; chưa đo frame lỗi, độ rõ âm thanh hoặc phụ đề burn bằng thị giác máy.",
        ]
    if key == "narrative_coherence":
        return [
            f"Tốc độ đọc phụ đề trung bình {float(signals.get('subtitle_mean_chars_per_second') or 0.0):.1f} ký tự/giây."
        ]
    return []


def _fallback_feedback(key: str, score: float, signals: dict) -> str:
    label = next(label for item_key, label, _ in _CRITERIA if item_key == key)
    if key == "translation_accuracy" and signals.get("translation_required"):
        return (
            "Chưa có kết luận ngữ nghĩa trực tiếp từ AI; điểm fallback được giới hạn thận trọng "
            f"ở {score:.1f}/100 từ độ phủ và timecode nguồn/đích."
        )
    if key in {"content_fidelity", "narrative_coherence"} and not signals.get("subtitles_expected"):
        return f"{label} không áp dụng vì job không yêu cầu dịch/phụ đề mới."
    if key == "av_subtitle_sync" and signals.get("subtitle_sync_source") == "not_applicable":
        return "Tiêu chí này không áp dụng: job không yêu cầu phụ đề hoặc giọng đọc mới; timeline video được giữ nguyên."
    if key == "av_subtitle_sync":
        level = "đạt" if score >= _PASS_SCORE[key] else "chưa đạt"
        if signals.get("tts_manifest_matched"):
            return (
                f"Đồng bộ cue/timecode {level} theo số đo hậu kỳ ({score:.1f}/100). "
                "Artifact speech khớp đủ từng cue và duration tổng; chưa có phép đo lip-sync hoặc word-level."
            )
        return (
            f"Đồng bộ cue/timecode {level} ở mức thận trọng ({score:.1f}/100). "
            "Artifact speech per-cue thiếu/không khớp nên chưa xác minh từng câu đọc; "
            "không có phép đo lip-sync hoặc word-level."
        )
    if key == "technical_quality":
        return (
            f"Chất lượng container/duration/stream {('đạt' if score >= _PASS_SCORE[key] else 'chưa đạt')} "
            f"theo số đo hậu kỳ ({score:.1f}/100). Chưa phân tích frame lỗi, độ rõ âm thanh hoặc phụ đề burn bằng thị giác máy."
        )
    level = "đạt" if score >= _PASS_SCORE[key] else "chưa đạt"
    return f"{label} {level} theo số đo hậu kỳ ({score:.1f}/100)."


def _source_target_timeline_score(
    source_events: list[SubtitleEvent],
    target_events: list[SubtitleEvent],
) -> tuple[float, float | None]:
    if not source_events or not target_events:
        return 0.0, None

    offsets: list[float] = []
    for source in source_events:
        matches = _overlapping_events(source, target_events)
        if not matches:
            nearest = _nearest_event(source, target_events)
            matches = [nearest] if nearest else []
        if matches:
            offsets.append(
                (
                    abs(source.start - min(item.start for item in matches))
                    + abs(source.end - max(item.end for item in matches))
                )
                / 2.0
            )
    mean_offset = sum(offsets) / max(len(offsets), 1)
    intersection = _interval_intersection_duration(source_events, target_events)
    source_union = _interval_union_duration(source_events)
    target_union = _interval_union_duration(target_events)
    coverage_score = 100.0 * (
        (intersection / max(source_union, 0.01)) * 0.5
        + (intersection / max(target_union, 0.01)) * 0.5
    )
    offset_score = _clamp(100.0 - max(0.0, mean_offset - 0.05) * 80.0)
    return _clamp(coverage_score * 0.65 + offset_score * 0.35), mean_offset


def _aligned_semantic_pairs(
    source_events: list[SubtitleEvent],
    target_events: list[SubtitleEvent],
) -> list[dict]:
    """Align changed cue segmentation by time overlap, never by list index."""

    result: list[dict] = []
    used_target_ids: set[int] = set()
    for index, source in enumerate(source_events, start=1):
        indexed_matches = [
            (target_index, target)
            for target_index, target in enumerate(target_events)
            if _event_overlap(source, target) > 0.01
        ]
        if not indexed_matches:
            nearest = _nearest_event(source, target_events)
            if nearest is not None:
                indexed_matches = [(target_events.index(nearest), nearest)]
        for target_index, _ in indexed_matches:
            used_target_ids.add(target_index)
        matches = [item for _, item in indexed_matches]
        result.append(
            {
                "index": index,
                "source": source.text,
                "translation": " ".join(item.text for item in matches),
                "source_time": [source.start, source.end],
                "target_time": (
                    [min(item.start for item in matches), max(item.end for item in matches)]
                    if matches
                    else None
                ),
            }
        )
    for target_index, target in enumerate(target_events):
        if target_index in used_target_ids:
            continue
        result.append(
            {
                "index": len(result) + 1,
                "source": "",
                "translation": target.text,
                "source_time": None,
                "target_time": [target.start, target.end],
            }
        )
    return result


def _overlapping_events(
    event: SubtitleEvent,
    candidates: list[SubtitleEvent],
) -> list[SubtitleEvent]:
    return [item for item in candidates if _event_overlap(event, item) > 0.01]


def _nearest_event(
    event: SubtitleEvent,
    candidates: list[SubtitleEvent],
) -> SubtitleEvent | None:
    if not candidates:
        return None
    midpoint = (event.start + event.end) / 2.0
    return min(candidates, key=lambda item: abs((item.start + item.end) / 2.0 - midpoint))


def _event_overlap(left: SubtitleEvent, right: SubtitleEvent) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def _interval_union_duration(events: list[SubtitleEvent]) -> float:
    if not events:
        return 0.0
    intervals = sorted((item.start, item.end) for item in events if item.end > item.start)
    if not intervals:
        return 0.0
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _interval_intersection_duration(
    left_events: list[SubtitleEvent],
    right_events: list[SubtitleEvent],
) -> float:
    boundaries = sorted(
        {
            boundary
            for item in [*left_events, *right_events]
            for boundary in (item.start, item.end)
        }
    )
    total = 0.0
    for start, end in zip(boundaries, boundaries[1:]):
        midpoint = (start + end) / 2.0
        in_left = any(item.start <= midpoint < item.end for item in left_events)
        in_right = any(item.start <= midpoint < item.end for item in right_events)
        if in_left and in_right:
            total += end - start
    return total


def _tts_segment_metrics(
    work_dir: Path,
    target_events: list[SubtitleEvent],
    narration_duration: float,
) -> tuple[int, int, float, bool, int]:
    aligned_dir = work_dir / "tts_aligned"
    actual_indices: set[int] = set()
    actual_durations: dict[int, float | None] = {}
    if aligned_dir.is_dir():
        for path in aligned_dir.glob("*_speech.wav"):
            if not path.is_file() or path.stat().st_size <= 0:
                continue
            match = re.fullmatch(r"(\d+)_speech", path.stem)
            if match:
                index = int(match.group(1))
                actual_indices.add(index)
                actual_durations[index] = _wav_duration(path)

    expected_slots: dict[int, float]
    try:
        from app.services.ai.voice import _voice_timeline_events, _voice_timeline_specs

        timeline_events = _voice_timeline_events(target_events, narration_duration)
        specs, _ = _voice_timeline_specs(timeline_events, narration_duration)
        expected_slots = {int(item[0]): float(item[2]) for item in specs}
    except Exception:
        expected_slots = {
            index: max(0.03, event.end - event.start)
            for index, event in enumerate(target_events, start=1)
        }

    expected_indices = set(expected_slots)
    if not expected_indices:
        return len(actual_indices), 0, (100.0 if not actual_indices else 0.0), not actual_indices, 0
    unreadable = {
        index
        for index in actual_indices & expected_indices
        if actual_durations.get(index) is None
    }
    overrun = {
        index
        for index in actual_indices & expected_indices
        if actual_durations.get(index) is not None
        and float(actual_durations[index]) > expected_slots[index] + 0.08
    }
    verified_indices = (actual_indices & expected_indices) - unreadable - overrun
    coverage = 100.0 * len(verified_indices) / len(expected_indices)
    matched = actual_indices == expected_indices and not unreadable and not overrun
    return len(actual_indices), len(expected_indices), _clamp(coverage), matched, len(overrun)


def _wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            return handle.getnframes() / frame_rate if frame_rate > 0 else None
    except (OSError, EOFError, wave.Error):
        return None


async def _probe_duration(ffmpeg: str, path: Path | None) -> float:
    if path is None or not path.is_file() or path.stat().st_size <= 0:
        return 0.0
    try:
        return await probe_video_duration(ffmpeg, path)
    except Exception:
        return 0.0


async def _probe_streams(
    ffmpeg: str,
    path: Path | None,
) -> tuple[bool | None, bool | None]:
    if path is None or not path.is_file() or path.stat().st_size <= 0:
        return None, None

    def probe() -> tuple[bool | None, bool | None]:
        try:
            completed = subprocess.run(
                [ffmpeg, "-hide_banner", "-i", str(path)],
                capture_output=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None, None
        output = b"\n".join([completed.stdout, completed.stderr]).decode(errors="ignore").lower()
        if not output:
            return None, None
        return "video:" in output, "audio:" in output

    return await asyncio.to_thread(probe)


def _read_subtitles(path: Path | None) -> list[SubtitleEvent]:
    if path is None or not path.is_file():
        return []
    try:
        return parse_srt(path)
    except Exception:
        return []


def _first_file(paths) -> Path | None:
    return next((path for path in paths if path.is_file() and path.stat().st_size > 0), None)


def _sample_indices(total: int, limit: int) -> list[int]:
    if total <= 0:
        return []
    if total <= limit:
        return list(range(total))
    if limit <= 1:
        return [total // 2]
    return sorted({round(index * (total - 1) / (limit - 1)) for index in range(limit)})


def _alignment_key(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text or "")).casefold()
    return "".join(char for char in normalized if char.isalnum())


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
    return AsyncOpenAI(
        api_key=settings.ninerouter_api_key or "local-ninerouter",
        base_url=settings.ninerouter_api_url,
    )
