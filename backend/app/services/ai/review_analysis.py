from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import subprocess
import unicodedata
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from openai import AsyncOpenAI

from app.core import settings
from app.models.job import ProcessingMode
from app.models.review import (
    AnalyzedScene,
    EditDecision,
    NarrationSegment,
    QualityIssue,
    RequiredVisuals,
    ReviewEvidence,
    ReviewKeyframe,
    ReviewQualityReport,
    SceneCandidate,
    StoryEvent,
    VerifiedReviewPackage,
)
from app.services.media.ffmpeg import find_ffmpeg, probe_video_duration
from app.services.presets import get_processing_profile
from app.services.subtitles.timing import SubtitleEvent, parse_srt


ProgressCallback = Callable[[str, int], None]


async def build_verified_review_package(
    video_path: Path,
    transcript_file: Path,
    work_dir: Path,
    duration: float,
    target_minutes: int,
    style: str,
    notes: str | None,
    on_progress: ProgressCallback,
    processing_mode: ProcessingMode | str | None = None,
) -> VerifiedReviewPackage:
    """Build a review from visual evidence before any narration is rendered.

    Every generated sentence is linked to a verified event and an EDL candidate.
    The source is sampled from beginning to end; no fixed four-frame shortcut is
    used for long movies.
    """

    events = parse_srt(transcript_file)
    profile = get_processing_profile(processing_mode)
    scene_dir = work_dir / "review_scenes"
    on_progress("Phát hiện shot và lấy keyframe toàn bộ phim", 53)
    scenes = await asyncio.to_thread(
        _detect_scenes_and_keyframes,
        video_path,
        scene_dir,
        duration,
        events,
        profile.review_max_scenes,
        profile.review_keyframes_per_scene,
    )
    if not scenes:
        raise RuntimeError("Không phát hiện được scene/keyframe hợp lệ trong video.")

    on_progress("AI đa phương thức phân tích hình ảnh, thoại và chữ trên màn hình", 60)
    scenes = await _analyze_scene_batches(
        scenes,
        events,
        on_progress,
        batch_size=profile.review_scene_batch_size,
        concurrency=profile.review_ai_concurrency,
    )
    scenes = _reconcile_character_identities(scenes)

    on_progress("Lập timeline sự kiện và kiểm tra danh tính nhân vật", 71)
    story_events = await _build_event_timeline(scenes)
    if not story_events:
        raise RuntimeError("Không tạo được timeline sự kiện có bằng chứng.")

    on_progress("Viết lời review từ các sự kiện đã kiểm chứng", 76)
    metadata, segments = await _write_verified_narration(
        story_events,
        scenes,
        target_minutes,
        style,
        notes,
    )
    segments = _normalize_segments(segments, story_events, target_minutes)
    if not segments:
        raise RuntimeError("AI không tạo được câu review gắn với event hợp lệ.")

    on_progress("Ghép từng câu với cảnh và tạo EDL", 79)
    decisions = _match_segments_to_scenes(segments, story_events, scenes)
    decisions = await _multimodal_rescore_selected_clips(
        segments,
        scenes,
        decisions,
        batch_size=profile.review_scene_batch_size,
    )
    decisions, changed = _replace_low_pre_render_matches(decisions)
    if changed:
        decisions = await _multimodal_rescore_selected_clips(
            segments,
            scenes,
            decisions,
            batch_size=profile.review_scene_batch_size,
        )
    report = _quality_report(segments, story_events, scenes, decisions)

    narration_script = " ".join(item.narration.strip() for item in segments if item.narration.strip())
    package = VerifiedReviewPackage(
        title=str(metadata.get("title") or "Bản review phim đã kiểm chứng"),
        target_minutes=target_minutes,
        hook=str(metadata.get("hook") or segments[0].narration),
        summary=str(metadata.get("summary") or " ".join(event.summary for event in story_events[:4])),
        narration_script=narration_script,
        thumbnail_text=str(metadata.get("thumbnail_text") or "CÂU CHUYỆN KHÔNG AI NGỜ"),
        tags=_clean_tags(metadata.get("tags")),
        scenes=scenes,
        events=story_events,
        narration_segments=segments,
        edit_decision_list=decisions,
        quality_report=report,
    )
    package.artifact_paths = _write_artifacts(package, work_dir)
    return package


def _detect_scenes_and_keyframes(
    video_path: Path,
    scene_dir: Path,
    duration: float,
    transcript_events: list[SubtitleEvent],
    max_scenes: int | None = None,
    keyframes_per_scene: int | None = None,
) -> list[AnalyzedScene]:
    scene_dir.mkdir(parents=True, exist_ok=True)
    for old in scene_dir.glob("*.jpg"):
        old.unlink(missing_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    actual_duration = duration or (
        float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(float(cap.get(cv2.CAP_PROP_FPS)), 1.0)
    )
    if actual_duration <= 0:
        cap.release()
        return []

    # Scan the whole source. Longer movies use a slightly larger interval, but
    # still keep cut boundaries and a 15-second maximum before consolidation.
    interval = 0.35 if actual_duration <= 120 else 0.75 if actual_duration <= 900 else 1.0 if actual_duration <= 3600 else 1.5
    min_scene = 0.8
    max_scene = 10.0
    boundaries = [0.0]
    previous_gray: np.ndarray | None = None
    previous_hist: np.ndarray | None = None
    last_boundary = 0.0
    timestamp = 0.0
    while timestamp < actual_duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok:
            timestamp += interval
            continue
        small = cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        elapsed = timestamp - last_boundary
        is_cut = False
        if previous_gray is not None and previous_hist is not None and elapsed >= min_scene:
            difference = float(np.mean(cv2.absdiff(gray, previous_gray))) / 255.0
            correlation = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_CORREL))
            is_cut = difference >= 0.18 or correlation <= 0.48
        if elapsed >= max_scene or is_cut:
            boundaries.append(round(timestamp, 3))
            last_boundary = timestamp
        previous_gray = gray
        previous_hist = hist
        timestamp += interval
    if actual_duration - boundaries[-1] < 1.0 and len(boundaries) > 1:
        boundaries[-1] = actual_duration
    else:
        boundaries.append(actual_duration)

    raw_ranges = [
        (boundaries[index], boundaries[index + 1])
        for index in range(len(boundaries) - 1)
        if boundaries[index + 1] - boundaries[index] >= 0.8
    ]
    scene_limit = settings.review_max_scenes if max_scenes is None else max_scenes
    ranges = _consolidate_ranges(raw_ranges, max(12, scene_limit))
    scenes: list[AnalyzedScene] = []
    keyframe_count = (
        max(3, settings.review_keyframes_per_scene)
        if keyframes_per_scene is None
        else max(1, keyframes_per_scene)
    )
    ratios = np.linspace(0.14, 0.86, keyframe_count).tolist()

    for index, (start, end) in enumerate(ranges, start=1):
        scene_id = f"scene_{index:04d}"
        keyframes: list[ReviewKeyframe] = []
        black_votes = 0
        for frame_index, ratio in enumerate(ratios, start=1):
            frame_time = start + (end - start) * float(ratio)
            cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            if float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) < 8.0:
                black_votes += 1
            height, width = frame.shape[:2]
            if width > 960:
                scale = 960 / width
                frame = cv2.resize(frame, (960, max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
            output = scene_dir / f"{scene_id}_{frame_index}.jpg"
            cv2.imwrite(str(output), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
            keyframes.append(ReviewKeyframe(time=round(frame_time, 3), path=str(output)))

        overlap = _subtitle_overlap(transcript_events, start, end)
        dialogue_text = " ".join(item.text for item in overlap)
        scenes.append(
            AnalyzedScene(
                scene_id=scene_id,
                start_time=round(start, 3),
                end_time=round(end, 3),
                keyframes=keyframes,
                thumbnail_path=keyframes[len(keyframes) // 2].path if keyframes else None,
                dialogue_summary=dialogue_text[:900],
                black_frame=bool(keyframes and black_votes >= math.ceil(len(keyframes) / 2)),
                evidence=ReviewEvidence(dialogue=[dialogue_text] if dialogue_text else []),
            )
        )
    cap.release()
    return scenes


def _consolidate_ranges(ranges: list[tuple[float, float]], limit: int) -> list[tuple[float, float]]:
    if len(ranges) <= limit:
        return ranges
    group_size = len(ranges) / limit
    consolidated: list[tuple[float, float]] = []
    for index in range(limit):
        left = min(len(ranges) - 1, int(round(index * group_size)))
        right = min(len(ranges), max(left + 1, int(round((index + 1) * group_size))))
        consolidated.append((ranges[left][0], ranges[right - 1][1]))
    return consolidated


async def _analyze_scene_batches(
    scenes: list[AnalyzedScene],
    transcript_events: list[SubtitleEvent],
    on_progress: ProgressCallback,
    *,
    batch_size: int | None = None,
    concurrency: int = 1,
) -> list[AnalyzedScene]:
    batch_size = max(1, min(8, batch_size or settings.review_scene_batch_size))
    batches = [scenes[index : index + batch_size] for index in range(0, len(scenes), batch_size)]
    semaphore = asyncio.Semaphore(max(1, min(3, concurrency)))

    async def analyze(index: int, batch: list[AnalyzedScene]) -> tuple[int, list[AnalyzedScene]]:
        async with semaphore:
            try:
                analyzed = await _analyze_scene_batch(batch, transcript_events)
            except Exception as exc:
                print(f"Review scene analysis fallback for batch {index + 1}: {exc}")
                analyzed = batch
            by_id = {scene.scene_id: scene for scene in analyzed}
            return index, [by_id.get(scene.scene_id, scene) for scene in batch]

    tasks = [asyncio.create_task(analyze(index, batch)) for index, batch in enumerate(batches)]
    indexed_results: dict[int, list[AnalyzedScene]] = {}
    completed = 0
    for task in asyncio.as_completed(tasks):
        index, analyzed = await task
        indexed_results[index] = analyzed
        completed += 1
        scene_count = min(completed * batch_size, len(scenes))
        percent = 60 + int((completed / max(len(batches), 1)) * 10)
        on_progress(f"Đã phân tích {scene_count}/{len(scenes)} scene", percent)
    return [scene for index in range(len(batches)) for scene in indexed_results[index]]


async def _analyze_scene_batch(
    scenes: list[AnalyzedScene],
    transcript_events: list[SubtitleEvent],
) -> list[AnalyzedScene]:
    client = _client()
    prompt = (
        "Bạn là kỹ sư AI đa phương thức phân tích phim. Chỉ mô tả thứ có bằng chứng trong ảnh/thoại/chữ trên màn hình; "
        "không dùng kiến thức sẵn có về bộ phim. Nhận diện nhân vật nhất quán, nhưng nếu không chắc tên phải ghi UNKNOWN. "
        "Phân biệt người thực hiện và người nhận hành động. Đọc cả hard subtitle nhìn thấy trong frame (OCR). "
        "Không biến hồi tưởng/giấc mơ thành hiện tại. Trả JSON {scenes:[...]}; mỗi phần tử bắt buộc có scene_id, "
        "characters, location, visible_actions, dialogue_summary, event_summary, important_objects, emotion, confidence, "
        "temporal_mode, credits, evidence:{visual:[],dialogue:[],subtitle:[]}. Evidence phải ngắn và cụ thể."
    )
    content: list[dict] = [{"type": "text", "text": prompt}]
    for scene in scenes:
        overlap = _subtitle_overlap(transcript_events, scene.start_time, scene.end_time)
        transcript = " ".join(item.text for item in overlap)
        content.append(
            {
                "type": "text",
                "text": (
                    f"{scene.scene_id} | {scene.start_time:.2f}-{scene.end_time:.2f}s | "
                    f"ASR/subtitle có timestamp: {transcript[:1200] or '[không có]'}"
                ),
            }
        )
        for keyframe in scene.keyframes:
            path = Path(keyframe.path)
            if not path.exists():
                continue
            content.append({"type": "text", "text": f"{scene.scene_id} keyframe tại {keyframe.time:.2f}s"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_file_b64(path)}"},
                }
            )

    response = await client.chat.completions.create(
        model=settings.ai_model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
        temperature=0.1,
        timeout=240,
    )
    data = _loads_json(response.choices[0].message.content or "{}")
    returned = data.get("scenes") if isinstance(data, dict) else None
    if not isinstance(returned, list):
        return scenes
    originals = {scene.scene_id: scene for scene in scenes}
    analyzed: list[AnalyzedScene] = []
    for item in returned:
        if not isinstance(item, dict):
            continue
        scene_id = str(item.get("scene_id") or "")
        original = originals.get(scene_id)
        if original is None:
            continue
        visual = _string_list(_dict(item.get("evidence")).get("visual"))
        dialogue = _string_list(_dict(item.get("evidence")).get("dialogue"))
        subtitle = _string_list(_dict(item.get("evidence")).get("subtitle"))
        analyzed.append(
            original.model_copy(
                update={
                    "characters": _string_list(item.get("characters")),
                    "location": _safe_text(item.get("location"), "UNKNOWN"),
                    "visible_actions": _string_list(item.get("visible_actions")),
                    "dialogue_summary": _safe_text(item.get("dialogue_summary"), original.dialogue_summary),
                    "event_summary": _safe_text(item.get("event_summary"), "UNKNOWN"),
                    "important_objects": _string_list(item.get("important_objects")),
                    "emotion": _safe_text(item.get("emotion"), "UNKNOWN"),
                    "confidence": _confidence(item.get("confidence")) or (0.72 if visual or dialogue or subtitle else 0.35),
                    "temporal_mode": _safe_text(item.get("temporal_mode"), "present"),
                    "credits": bool(item.get("credits", False)),
                    "evidence": ReviewEvidence(visual=visual, dialogue=dialogue, subtitle=subtitle),
                }
            )
        )
    return analyzed or scenes


async def _build_event_timeline(scenes: list[AnalyzedScene]) -> list[StoryEvent]:
    usable = [scene for scene in scenes if not scene.black_frame and not scene.credits]
    if not usable:
        usable = scenes
    compact = [
        {
            "scene_id": scene.scene_id,
            "start": scene.start_time,
            "end": scene.end_time,
            "characters": scene.characters,
            "actions": scene.visible_actions,
            "objects": scene.important_objects,
            "event": scene.event_summary,
            "dialogue": scene.dialogue_summary[:300],
            "confidence": scene.confidence,
            "temporal_mode": scene.temporal_mode,
            "evidence": scene.evidence.model_dump(),
        }
        for scene in usable
    ]
    prompt = (
        "Từ scene đã phân tích, lập timeline sự kiện theo đúng source time. Không thêm sự kiện hay tên nhân vật ngoài evidence. "
        "Gộp scene liên tiếp cùng một sự kiện; giữ flashback/dream đúng loại; nêu quan hệ nguyên nhân-kết quả chỉ khi có bằng chứng. "
        "Mỗi event phải có >=1 scene_id thật. Trả JSON {events:[{event_id,order_index,scene_ids,characters,summary,cause,"
        "consequence,confidence,verification_status}]}. verification_status chỉ là verified khi có evidence trực tiếp, còn lại unverified."
    )
    try:
        response = await _client().chat.completions.create(
            model=settings.ai_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(compact, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            timeout=240,
        )
        data = _loads_json(response.choices[0].message.content or "{}")
        raw_events = data.get("events", []) if isinstance(data, dict) else []
    except Exception as exc:
        print(f"Review event timeline fallback: {exc}")
        raw_events = []
    timeline = _validate_event_timeline(raw_events, usable)
    return timeline or _fallback_event_timeline(usable)


def _reconcile_character_identities(scenes: list[AnalyzedScene]) -> list[AnalyzedScene]:
    """Keep animated characters stable when a later batch emits UNKNOWN_* aliases."""

    descriptor_map: dict[str, str] = {}
    for scene in scenes:
        for character in scene.characters:
            upper = character.upper()
            if upper.startswith("UNKNOWN"):
                continue
            tokens = _identity_descriptors(character)
            for token in tokens:
                descriptor_map.setdefault(token, character)

    reconciled: list[AnalyzedScene] = []
    for scene in scenes:
        replacements: dict[str, str] = {}
        characters: list[str] = []
        for character in scene.characters:
            canonical = character
            if character.upper().startswith("UNKNOWN"):
                candidates = [descriptor_map[token] for token in _identity_descriptors(character) if token in descriptor_map]
                if candidates:
                    canonical = candidates[0]
                    replacements[character] = canonical
            if canonical not in characters:
                characters.append(canonical)
        actions = list(scene.visible_actions)
        summary = scene.event_summary
        dialogue = scene.dialogue_summary
        for alias, canonical in replacements.items():
            actions = [value.replace(alias, canonical) for value in actions]
            summary = summary.replace(alias, canonical)
            dialogue = dialogue.replace(alias, canonical)
        reconciled.append(
            scene.model_copy(
                update={
                    "characters": characters,
                    "visible_actions": actions,
                    "event_summary": summary,
                    "dialogue_summary": dialogue,
                }
            )
        )
    return reconciled


def _identity_descriptors(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", value.lower()).replace("_", "-")
    words = set(re.findall(r"[\w]+", normalized, flags=re.UNICODE))
    descriptors: set[str] = set()
    color_aliases = {
        "white": ("white", "trang", "bạch"),
        "red": ("red", "đỏ", "hong"),
        "black": ("black", "đen"),
        "brown": ("brown", "nâu"),
        "blonde": ("blonde", "yellow", "vàng"),
        "blue": ("blue", "xanh"),
    }
    for canonical, aliases in color_aliases.items():
        if any(alias in words for alias in aliases):
            descriptors.add(f"hair:{canonical}")
    return descriptors


def _validate_event_timeline(raw_events: object, scenes: list[AnalyzedScene]) -> list[StoryEvent]:
    if not isinstance(raw_events, list):
        return []
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    accepted: list[StoryEvent] = []
    for item in raw_events:
        if not isinstance(item, dict):
            continue
        ids = [value for value in _string_list(item.get("scene_ids")) if value in scene_by_id]
        if not ids:
            continue
        linked = sorted((scene_by_id[value] for value in ids), key=lambda scene: scene.start_time)
        evidence_count = sum(
            len(scene.evidence.visual) + len(scene.evidence.dialogue) + len(scene.evidence.subtitle)
            for scene in linked
        )
        claimed_status = str(item.get("verification_status") or "unverified").lower()
        status = "verified" if claimed_status == "verified" and evidence_count > 0 else "unverified"
        accepted.append(
            StoryEvent(
                event_id="pending",
                order_index=0,
                start_time=min(scene.start_time for scene in linked),
                end_time=max(scene.end_time for scene in linked),
                scene_ids=[scene.scene_id for scene in linked],
                characters=_string_list(item.get("characters")),
                summary=_safe_text(item.get("summary"), linked[0].event_summary),
                cause=_safe_text(item.get("cause"), ""),
                consequence=_safe_text(item.get("consequence"), ""),
                confidence=_confidence(item.get("confidence")),
                evidence_count=evidence_count,
                verification_status=status,
            )
        )
    accepted.sort(key=lambda event: event.start_time)
    return [
        event.model_copy(update={"event_id": f"event_{index:04d}", "order_index": index})
        for index, event in enumerate(accepted, start=1)
    ]


def _fallback_event_timeline(scenes: list[AnalyzedScene]) -> list[StoryEvent]:
    events: list[StoryEvent] = []
    for index, scene in enumerate(scenes, start=1):
        evidence_count = len(scene.evidence.visual) + len(scene.evidence.dialogue) + len(scene.evidence.subtitle)
        events.append(
            StoryEvent(
                event_id=f"event_{index:04d}",
                order_index=index,
                start_time=scene.start_time,
                end_time=scene.end_time,
                scene_ids=[scene.scene_id],
                characters=scene.characters,
                summary=scene.event_summary if scene.event_summary != "UNKNOWN" else scene.dialogue_summary[:240],
                confidence=scene.confidence,
                evidence_count=evidence_count,
                verification_status="verified" if evidence_count else "unverified",
            )
        )
    return events


async def _write_verified_narration(
    events: list[StoryEvent],
    scenes: list[AnalyzedScene],
    target_minutes: int,
    style: str,
    notes: str | None,
) -> tuple[dict, list[NarrationSegment]]:
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    event_data = []
    for event in events:
        linked = [scene_by_id[value] for value in event.scene_ids if value in scene_by_id]
        event_data.append(
            {
                **event.model_dump(),
                "scene_facts": [
                    {
                        "scene_id": scene.scene_id,
                        "characters": scene.characters,
                        "actions": scene.visible_actions,
                        "objects": scene.important_objects,
                        "location": scene.location,
                        "summary": scene.event_summary,
                        "evidence": scene.evidence.model_dump(),
                    }
                    for scene in linked
                ],
            }
        )
    segment_target = max(8, min(len(events), target_minutes * 9))
    word_target = target_minutes * 145
    prompt = (
        "Bạn viết review phim tiếng Việt chỉ từ EVENT TIMELINE đã kiểm chứng dưới đây. Cấm dùng kiến thức ngoài dữ liệu, cấm đổi tên, "
        "đổi người hành động, đảo thời gian hoặc tiết lộ sự kiện chưa tới. Mỗi segment chỉ một câu dễ đọc, phải gắn một event_id thật "
        "và required_visuals cụ thể nhìn thấy được. Bỏ event unverified/UNKNOWN. Lời kể theo source time tăng dần. "
        "Các giá trị trong required_visuals phải copy nguyên văn từ scene_facts tương ứng, không dịch sang ngôn ngữ khác. "
        f"Mục tiêu khoảng {segment_target} câu và tối đa {word_target} từ; phong cách {style}. "
        "Trả JSON gồm title,hook,summary,thumbnail_text,tags,segments. Mỗi segment: event_id,narration,required_visuals"
        "{characters,actions,objects,locations},forbidden_visuals,candidate_scene_ids,confidence,purpose."
    )
    if notes:
        prompt += f" Ghi chú người dùng (chỉ áp dụng nếu không mâu thuẫn evidence): {notes[:1500]}"
    try:
        response = await _client().chat.completions.create(
            model=settings.ai_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(event_data, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=240,
        )
        data = _loads_json(response.choices[0].message.content or "{}")
    except Exception as exc:
        print(f"Verified review narration fallback: {exc}")
        data = {}
    raw_segments = data.get("segments", []) if isinstance(data, dict) else []
    valid_events = {event.event_id: event for event in events}
    segments: list[NarrationSegment] = []
    if isinstance(raw_segments, list):
        for item in raw_segments:
            if not isinstance(item, dict):
                continue
            event_id = str(item.get("event_id") or "")
            event = valid_events.get(event_id)
            narration = unicodedata.normalize("NFC", _safe_text(item.get("narration"), ""))
            if event is None or not narration or event.verification_status != "verified":
                continue
            required = _dict(item.get("required_visuals"))
            candidate_ids = [value for value in _string_list(item.get("candidate_scene_ids")) if value in event.scene_ids]
            segments.append(
                NarrationSegment(
                    segment_id="pending",
                    event_id=event_id,
                    narration=narration,
                    required_visuals=RequiredVisuals(
                        characters=_string_list(required.get("characters")),
                        actions=_string_list(required.get("actions")),
                        objects=_string_list(required.get("objects")),
                        locations=_string_list(required.get("locations")),
                    ),
                    forbidden_visuals=_string_list(item.get("forbidden_visuals")),
                    candidate_scene_ids=candidate_ids or event.scene_ids,
                    estimated_voice_duration=_estimated_voice_duration(narration),
                    confidence=_confidence(item.get("confidence")),
                    purpose=_safe_text(item.get("purpose"), "Minh họa sự kiện đã kiểm chứng"),
                )
            )
    if not segments:
        for event in events:
            if event.verification_status != "verified" or not event.summary.strip():
                continue
            narration = unicodedata.normalize("NFC", event.summary.strip())
            segments.append(
                NarrationSegment(
                    segment_id="pending",
                    event_id=event.event_id,
                    narration=narration,
                    required_visuals=RequiredVisuals(characters=event.characters),
                    candidate_scene_ids=event.scene_ids,
                    estimated_voice_duration=_estimated_voice_duration(narration),
                    confidence=event.confidence,
                )
            )
    metadata = data if isinstance(data, dict) else {}
    return metadata, segments


def _normalize_segments(
    segments: list[NarrationSegment],
    events: list[StoryEvent],
    target_minutes: int,
) -> list[NarrationSegment]:
    order = {event.event_id: event.order_index for event in events}
    normalized: list[NarrationSegment] = []
    for segment in sorted(segments, key=lambda item: order.get(item.event_id, 999999)):
        # The inspector edits one narration sentence at a time. Split multi-sentence
        # responses without changing their evidence/event binding.
        sentences = [
            value.strip()
            for value in re.split(r"(?<=[.!?…])\s+", segment.narration)
            if value.strip()
        ] or [segment.narration.strip()]
        for sentence in sentences:
            normalized.append(
                segment.model_copy(
                    update={
                        "segment_id": f"segment_{len(normalized) + 1:04d}",
                        "narration": sentence,
                        "estimated_voice_duration": _estimated_voice_duration(sentence),
                    }
                )
            )
    max_segments = max(12, target_minutes * 12)
    return normalized[:max_segments]


def _match_segments_to_scenes(
    segments: list[NarrationSegment],
    events: list[StoryEvent],
    scenes: list[AnalyzedScene],
) -> list[EditDecision]:
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    event_by_id = {event.event_id: event for event in events}
    decisions: list[EditDecision] = []
    cursor = 0.0
    recent: list[str] = []
    for segment in segments:
        event = event_by_id[segment.event_id]
        ids: list[str] = []
        for scene_id in segment.candidate_scene_ids + event.scene_ids:
            if scene_id in scene_by_id and scene_id not in ids:
                ids.append(scene_id)
        candidates: list[SceneCandidate] = []
        for scene_id in ids:
            scene = scene_by_id[scene_id]
            score, reason = _scene_match_score(segment, event, scene)
            if scene_id in recent[-2:]:
                reason += "; cảnh vừa dùng nên chỉ lặp lại khi không có phương án tương đương"
            clip_duration = min(7.0, max(2.0, segment.estimated_voice_duration))
            start = min(
                max(scene.start_time, scene.start_time + 0.15),
                max(scene.start_time, scene.end_time - clip_duration),
            )
            end = min(scene.end_time, start + clip_duration)
            if end - start < 2.0:
                start = max(scene.start_time, end - 2.0)
            candidates.append(
                SceneCandidate(
                    candidate_id=f"{segment.segment_id}:{scene_id}",
                    scene_id=scene_id,
                    start_seconds=round(start, 3),
                    end_seconds=round(max(start + 0.8, end), 3),
                    thumbnail_path=scene.thumbnail_path,
                    match_score=round(score, 4),
                    match_reason=reason,
                )
            )
        candidates.sort(key=lambda item: item.match_score, reverse=True)
        if not candidates:
            continue
        best_score = candidates[0].match_score
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate.scene_id not in recent[-2:] and candidate.match_score >= best_score - 0.06
            ),
            candidates[0],
        )
        recent.append(selected.scene_id)
        voice_start = cursor
        voice_end = cursor + segment.estimated_voice_duration
        cursor = voice_end
        decisions.append(
            EditDecision(
                segment_id=segment.segment_id,
                event_id=segment.event_id,
                narration=segment.narration,
                voice_start=round(voice_start, 3),
                voice_end=round(voice_end, 3),
                selected_candidate_id=selected.candidate_id,
                source_clips=[selected],
                alternatives=candidates[:3],
            )
        )
    return decisions


def _scene_match_score(
    segment: NarrationSegment,
    event: StoryEvent,
    scene: AnalyzedScene,
) -> tuple[float, str]:
    required = segment.required_visuals
    scene_text = " ".join(
        scene.characters
        + scene.visible_actions
        + scene.important_objects
        + [scene.location, scene.event_summary, scene.dialogue_summary]
    )
    narration_overlap = _token_overlap(segment.narration, scene_text)
    character = _list_overlap(required.characters, scene.characters + [scene_text], default=0.75)
    action = _list_overlap(required.actions, scene.visible_actions + [scene_text], default=0.72)
    objects = _list_overlap(required.objects, scene.important_objects + [scene_text], default=0.72)
    location = _list_overlap(required.locations, [scene.location, scene_text], default=0.75)
    event_link = 1.0 if scene.scene_id in event.scene_ids else 0.0
    evidence = min(1.0, (
        len(scene.evidence.visual) + len(scene.evidence.dialogue) + len(scene.evidence.subtitle)
    ) / 3.0)
    score = (
        narration_overlap * 0.16
        + character * 0.22
        + action * 0.18
        + objects * 0.12
        + location * 0.07
        + event_link * 0.15
        + scene.confidence * 0.05
        + evidence * 0.05
    )
    score = min(1.0, max(0.0, score))
    matched: list[str] = []
    if character >= 0.75 and required.characters:
        matched.append("đúng nhân vật")
    if action >= 0.65 and required.actions:
        matched.append("đúng hành động")
    if objects >= 0.65 and required.objects:
        matched.append("đúng vật thể")
    if location >= 0.65 and required.locations:
        matched.append("đúng bối cảnh")
    if event_link:
        matched.append("cùng event đã kiểm chứng")
    if evidence:
        matched.append("có bằng chứng hình/thoại/OCR")
    return score, ", ".join(matched) or "Khớp ngữ nghĩa tổng quát; cần kiểm tra thủ công"


def _quality_report(
    segments: list[NarrationSegment],
    events: list[StoryEvent],
    scenes: list[AnalyzedScene],
    decisions: list[EditDecision],
) -> ReviewQualityReport:
    scores = [item.source_clips[0].match_score for item in decisions if item.source_clips]
    direct = 100.0 * sum(score >= 0.75 for score in scores) / max(len(segments), 1)
    event_order = {event.event_id: event.order_index for event in events}
    order_values = [event_order.get(item.event_id, 999999) for item in decisions]
    chronology = 100.0 if all(a <= b for a, b in zip(order_values, order_values[1:])) else 60.0
    evidence = 100.0 * sum(event.verification_status == "verified" for event in events) / max(len(events), 1)
    named = [name for scene in scenes for name in scene.characters if name.upper() != "UNKNOWN"]
    unknown = [name for scene in scenes for name in scene.characters if name.upper() == "UNKNOWN"]
    character_score = 100.0 if not unknown else max(60.0, 100.0 - 4.0 * len(unknown) / max(len(named) + len(unknown), 1))
    overall = direct * 0.45 + chronology * 0.25 + evidence * 0.20 + character_score * 0.10
    issues: list[QualityIssue] = []
    for decision in decisions:
        if not decision.source_clips:
            issues.append(QualityIssue(severity="error", code="NO_CLIP", message="Câu chưa có cảnh minh họa.", segment_id=decision.segment_id))
        elif decision.source_clips[0].match_score < 0.75:
            issues.append(
                QualityIssue(
                    severity="error",
                    code="LOW_VISUAL_MATCH",
                    message=f"Điểm khớp cảnh {decision.source_clips[0].match_score:.2f} dưới 0.75.",
                    segment_id=decision.segment_id,
                )
            )
    if chronology < 100:
        issues.append(QualityIssue(severity="error", code="CHRONOLOGY", message="EDL có cảnh đảo thứ tự sự kiện."))
    if evidence < 95:
        issues.append(QualityIssue(severity="warning", code="UNVERIFIED_EVENT", message="Timeline còn event thiếu bằng chứng trực tiếp."))
    return ReviewQualityReport(
        overall_score=round(overall, 2),
        direct_visual_match_percent=round(direct, 2),
        chronology_score=round(chronology, 2),
        evidence_score=round(evidence, 2),
        character_consistency_score=round(character_score, 2),
        passed=overall >= settings.review_quality_threshold and not any(issue.severity == "error" for issue in issues),
        issues=issues,
    )


async def _multimodal_rescore_selected_clips(
    segments: list[NarrationSegment],
    scenes: list[AnalyzedScene],
    decisions: list[EditDecision],
    *,
    batch_size: int | None = None,
) -> list[EditDecision]:
    """Resolve cross-language action matching with the actual keyframe evidence."""

    segment_by_id = {item.segment_id: item for item in segments}
    scene_by_id = {item.scene_id: item for item in scenes}
    ai_scores: dict[str, tuple[float, str, float | None]] = {}
    batch_size = max(1, min(8, batch_size or settings.review_scene_batch_size))
    for offset in range(0, len(decisions), batch_size):
        batch = decisions[offset : offset + batch_size]
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "Bạn chấm semantic scene matching trước render. Với từng segment, xem narration tiếng Việt, "
                    "required_visuals và keyframe; chấm 0..1 xem cảnh có trực tiếp cho thấy đúng nhân vật, hành động, "
                    "đồ vật và bối cảnh hay không. Cấm dùng kiến thức phim ngoài ảnh/evidence. "
                    "Trả JSON {segments:[{segment_id,score,reason,best_keyframe_time}]}; dưới 0.75 nếu sai "
                    "chủ thể/hành động. best_keyframe_time phải là timestamp keyframe đã được ghi nhãn nơi bằng chứng rõ nhất."
                ),
            }
        ]
        valid_ids: list[str] = []
        for decision in batch:
            if not decision.source_clips:
                continue
            segment = segment_by_id.get(decision.segment_id)
            scene = scene_by_id.get(decision.source_clips[0].scene_id)
            if segment is None or scene is None:
                continue
            valid_ids.append(decision.segment_id)
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"{decision.segment_id}\nNARRATION: {decision.narration}\n"
                        f"REQUIRED: {segment.required_visuals.model_dump_json()}\n"
                        f"SCENE FACTS: {scene.model_dump_json(exclude={'keyframes', 'thumbnail_path'})}"
                    ),
                }
            )
            for keyframe in scene.keyframes[:3]:
                keyframe_path = Path(keyframe.path)
                if not keyframe_path.exists():
                    continue
                content.append({"type": "text", "text": f"Keyframe source {keyframe.time:.2f}s"})
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_file_b64(keyframe_path)}"},
                    }
                )
        if not valid_ids:
            continue
        try:
            response = await _client().chat.completions.create(
                model=settings.ai_model,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=180,
            )
            data = _loads_json(response.choices[0].message.content or "{}")
            values = data.get("segments", []) if isinstance(data, dict) else []
            if isinstance(values, list):
                for item in values:
                    if not isinstance(item, dict):
                        continue
                    segment_id = str(item.get("segment_id") or "")
                    if segment_id in valid_ids:
                        ai_scores[segment_id] = (
                            _confidence(item.get("score")),
                            _safe_text(item.get("reason"), "Gemini xác nhận bằng keyframe và scene facts."),
                            _optional_number(item.get("best_keyframe_time")),
                        )
        except Exception as exc:
            print(f"Pre-render multimodal rescoring fallback: {exc}")

    updated: list[EditDecision] = []
    for decision in decisions:
        value = ai_scores.get(decision.segment_id)
        if value is None or not decision.source_clips:
            updated.append(decision)
            continue
        ai_score, reason, best_time = value
        selected = decision.source_clips[0]
        score = min(1.0, selected.match_score * 0.25 + ai_score * 0.75)
        scene = scene_by_id.get(selected.scene_id)
        start_seconds = selected.start_seconds
        end_seconds = selected.end_seconds
        if scene is not None and best_time is not None and scene.start_time <= best_time <= scene.end_time:
            scene_span = max(0.8, scene.end_time - scene.start_time)
            window = min(2.5, max(0.7, scene_span * 0.30))
            start_seconds = max(scene.start_time, min(best_time - window / 2.0, scene.end_time - window))
            end_seconds = min(scene.end_time, start_seconds + window)
        selected = selected.model_copy(
            update={
                "start_seconds": round(start_seconds, 3),
                "end_seconds": round(end_seconds, 3),
                "match_score": round(score, 4),
                "match_reason": f"Gemini visual-semantic: {reason}",
            }
        )
        alternatives = [
            selected if item.candidate_id == selected.candidate_id else item
            for item in decision.alternatives
        ]
        updated.append(decision.model_copy(update={"source_clips": [selected], "alternatives": alternatives}))
    return updated


def _replace_low_pre_render_matches(decisions: list[EditDecision]) -> tuple[list[EditDecision], bool]:
    """Try the next strong candidate when visual QA rejects the initial pick."""

    changed = False
    updated: list[EditDecision] = []
    for decision in decisions:
        current = decision.source_clips[0] if decision.source_clips else None
        if current is None or current.match_score >= 0.75:
            updated.append(decision)
            continue
        alternative = next(
            (
                item
                for item in sorted(decision.alternatives, key=lambda value: value.match_score, reverse=True)
                if item.candidate_id != current.candidate_id and item.match_score >= 0.75
            ),
            None,
        )
        if alternative is None:
            updated.append(decision)
            continue
        updated.append(
            decision.model_copy(
                update={
                    "selected_candidate_id": alternative.candidate_id,
                    "source_clips": [alternative],
                }
            )
        )
        changed = True
    return updated, changed


def _write_artifacts(package: VerifiedReviewPackage, work_dir: Path) -> dict[str, str]:
    artifacts = {
        "scene_timeline": work_dir / "scene_timeline.json",
        "event_timeline": work_dir / "event_timeline.json",
        "verified_script": work_dir / "verified_review_script.json",
        "edit_decision_list": work_dir / "edit_decision_list.json",
        "quality_report": work_dir / "quality_report.json",
    }
    payloads = {
        "scene_timeline": [item.model_dump(mode="json") for item in package.scenes],
        "event_timeline": [item.model_dump(mode="json") for item in package.events],
        "verified_script": [item.model_dump(mode="json") for item in package.narration_segments],
        "edit_decision_list": [item.model_dump(mode="json") for item in package.edit_decision_list],
        "quality_report": package.quality_report.model_dump(mode="json"),
    }
    for key, path in artifacts.items():
        path.write_text(json.dumps(payloads[key], ensure_ascii=False, indent=2), encoding="utf-8")
    return {key: str(path) for key, path in artifacts.items()}


def rescale_edl_voice_timeline(decisions: list[EditDecision], actual_duration: float) -> list[EditDecision]:
    estimated = max((item.voice_end for item in decisions), default=0.0)
    if estimated <= 0 or actual_duration <= 0:
        return decisions
    scale = actual_duration / estimated
    result: list[EditDecision] = []
    cursor = 0.0
    for decision in decisions:
        duration = max(0.8, (decision.voice_end - decision.voice_start) * scale)
        result.append(decision.model_copy(update={"voice_start": round(cursor, 3), "voice_end": round(cursor + duration, 3)}))
        cursor += duration
    return result


async def verify_rendered_review(
    rendered_video: Path,
    package: VerifiedReviewPackage,
    work_dir: Path,
    *,
    processing_mode: ProcessingMode | str | None = None,
) -> ReviewQualityReport:
    """Check the actual rendered frames against every narration sentence."""

    qa_dir = work_dir / "review_qa_frames"
    qa_dir.mkdir(parents=True, exist_ok=True)
    for old in qa_dir.glob("*.jpg"):
        old.unlink(missing_ok=True)
    frame_paths, decoded_duration, black_segments = await asyncio.to_thread(
        _extract_render_qa_frames,
        rendered_video,
        package.edit_decision_list,
        qa_dir,
    )
    probed_duration = await probe_video_duration(find_ffmpeg() or "ffmpeg", rendered_video)
    media_duration = probed_duration if probed_duration > 0 else decoded_duration
    expected_duration = max((item.voice_end for item in package.edit_decision_list), default=0.0)
    segment_by_id = {item.segment_id: item for item in package.narration_segments}
    scores: dict[str, float] = {}
    notes: dict[str, str] = {}
    direct_matches: dict[str, bool] = {}
    decisions = package.edit_decision_list
    profile = get_processing_profile(processing_mode)
    batch_size = max(1, min(8, profile.review_scene_batch_size))
    for offset in range(0, len(decisions), batch_size):
        batch = decisions[offset : offset + batch_size]
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "Bạn làm QA video review. Mỗi ảnh là frame giữa lúc câu voice tương ứng đang đọc. "
                    "Chỉ chấm hình ảnh thực sự nhìn thấy có minh họa trực tiếp câu kể và required_visuals hay không; "
                    "bỏ qua chữ phụ đề review phủ trên ảnh. Không suy diễn từ kiến thức phim. "
                    "Trả JSON {segments:[{segment_id,direct_match,score,reason}]}, score 0..1. "
                    "direct_match=true và score >=0.75 khi chuỗi frame minh họa trực tiếp đúng câu; "
                    "direct_match=false và score <0.75 khi sai nhân vật/hành động/vật thể, frame đen hoặc không trực tiếp."
                ),
            }
        ]
        valid_ids: list[str] = []
        for decision in batch:
            paths = frame_paths.get(decision.segment_id, [])
            segment = segment_by_id.get(decision.segment_id)
            if not paths or segment is None:
                scores[decision.segment_id] = 0.0
                notes[decision.segment_id] = "Không lấy được frame QA."
                continue
            valid_ids.append(decision.segment_id)
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"{decision.segment_id}\nVOICE: {decision.narration}\n"
                        f"REQUIRED: {segment.required_visuals.model_dump_json()}"
                    ),
                }
            )
            for path in paths:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_file_b64(path)}"},
                    }
                )
        if not valid_ids:
            continue
        try:
            response = await _client().chat.completions.create(
                model=settings.ai_model,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=180,
            )
            data = _loads_json(response.choices[0].message.content or "{}")
            values = data.get("segments", []) if isinstance(data, dict) else []
            if isinstance(values, list):
                for item in values:
                    if not isinstance(item, dict):
                        continue
                    segment_id = str(item.get("segment_id") or "")
                    if segment_id not in valid_ids:
                        continue
                    direct_match = bool(item.get("direct_match"))
                    score = _confidence(item.get("score"))
                    if direct_match:
                        score = max(0.75, score)
                    else:
                        score = min(0.74, score)
                    scores[segment_id] = score
                    direct_matches[segment_id] = direct_match
                    notes[segment_id] = _safe_text(item.get("reason"), "Gemini không nêu lý do.")
        except Exception as exc:
            print(f"Post-render visual QA fallback: {exc}")
            for decision in batch:
                if decision.segment_id in valid_ids and decision.source_clips:
                    scores[decision.segment_id] = decision.source_clips[0].match_score
                    direct_matches[decision.segment_id] = decision.source_clips[0].match_score >= 0.75
                    notes[decision.segment_id] = "Dùng điểm pre-render vì Gemini QA lỗi kết nối."

    for segment_id in black_segments:
        scores[segment_id] = 0.0
        direct_matches[segment_id] = False
        notes[segment_id] = "Frame render bị đen hoặc gần như trống."

    direct = 100.0 * sum(scores.get(item.segment_id, 0.0) >= 0.75 for item in decisions) / max(len(decisions), 1)
    duration_error = abs(media_duration - expected_duration)
    sync_score = 100.0 if duration_error <= 0.5 else max(0.0, 100.0 - duration_error * 15.0)
    chronology = min(package.quality_report.chronology_score, sync_score)
    overall = (
        direct * 0.50
        + chronology * 0.25
        + package.quality_report.evidence_score * 0.15
        + package.quality_report.character_consistency_score * 0.10
    )
    issues: list[QualityIssue] = []
    for decision in decisions:
        score = scores.get(decision.segment_id, 0.0)
        if score < 0.75 or not direct_matches.get(decision.segment_id, score >= 0.75):
            issues.append(
                QualityIssue(
                    severity="error",
                    code="POST_RENDER_VISUAL_MISMATCH",
                    message=f"{notes.get(decision.segment_id, 'Hình chưa khớp trực tiếp')} (score {score:.2f}).",
                    segment_id=decision.segment_id,
                )
            )
    if duration_error > 0.5:
        issues.append(
            QualityIssue(
                severity="error",
                code="VOICE_VIDEO_DRIFT",
                message=f"Video lệch thời lượng voice {duration_error:.3f} giây.",
            )
        )
    report = ReviewQualityReport(
        phase="post_render",
        overall_score=round(overall, 2),
        direct_visual_match_percent=round(direct, 2),
        chronology_score=round(chronology, 2),
        evidence_score=package.quality_report.evidence_score,
        character_consistency_score=package.quality_report.character_consistency_score,
        passed=overall >= settings.review_quality_threshold and not issues,
        issues=issues,
    )
    (work_dir / "quality_report_post_render.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return report


async def verify_edited_narration(
    narration: str,
    event: StoryEvent,
    scenes: list[AnalyzedScene],
) -> tuple[bool, str, RequiredVisuals | None]:
    """Reject manual narration edits that introduce claims outside evidence."""

    linked = [scene for scene in scenes if scene.scene_id in event.scene_ids]
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Kiểm tra câu review có được chứng minh trực tiếp bởi event, thoại/OCR và frame hay không. "
                "Cấm chấp nhận tên, hành động, vật thể, quan hệ hoặc kết quả không có evidence. "
                "Trả JSON {valid:boolean,confidence:number,reason:string,required_visuals:{characters:[],actions:[],objects:[],locations:[]}}. "
                "required_visuals chỉ được chứa đúng người/hành động/vật thể/địa điểm mà CÂU REVIEW mới thực sự nhắc đến và evidence xác nhận; "
                "không giữ hành động từ câu cũ, không thêm chi tiết chỉ xuất hiện ở cảnh nhưng không có trong câu.\n"
                f"CÂU REVIEW: {narration}\nEVENT: {event.model_dump_json()}\n"
                f"SCENES: {json.dumps([scene.model_dump(exclude={'keyframes', 'thumbnail_path'}) for scene in linked], ensure_ascii=False)}"
            ),
        }
    ]
    for scene in linked[:4]:
        if not scene.thumbnail_path or not Path(scene.thumbnail_path).exists():
            continue
        content.append({"type": "text", "text": scene.scene_id})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_file_b64(Path(scene.thumbnail_path))}"},
            }
        )
    try:
        response = await _client().chat.completions.create(
            model=settings.ai_model,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=120,
        )
        data = _loads_json(response.choices[0].message.content or "{}")
        confidence = _confidence(data.get("confidence"))
        valid = bool(data.get("valid")) and confidence >= 0.7
        raw_visuals = _dict(data.get("required_visuals"))
        required_visuals = RequiredVisuals(
            characters=_string_list(raw_visuals.get("characters")),
            actions=_string_list(raw_visuals.get("actions")),
            objects=_string_list(raw_visuals.get("objects")),
            locations=_string_list(raw_visuals.get("locations")),
        )
        return (
            valid,
            _safe_text(data.get("reason"), "Gemini không nêu lý do."),
            required_visuals if valid else None,
        )
    except Exception as exc:
        return False, f"Không gọi được Gemini để kiểm chứng câu sửa: {exc}", None


def replace_failed_decisions_with_alternatives(
    decisions: list[EditDecision],
    report: ReviewQualityReport,
) -> tuple[list[EditDecision], bool]:
    failed_ids = {issue.segment_id for issue in report.issues if issue.segment_id}
    changed = False
    updated: list[EditDecision] = []
    for decision in decisions:
        if decision.segment_id not in failed_ids or len(decision.alternatives) < 2:
            updated.append(decision)
            continue
        current_id = decision.source_clips[0].candidate_id if decision.source_clips else ""
        alternative = next(
            (
                item
                for item in decision.alternatives
                if item.candidate_id != current_id and item.match_score >= 0.75
            ),
            None,
        )
        if alternative is None:
            updated.append(decision)
            continue
        updated.append(
            decision.model_copy(
                update={
                    "selected_candidate_id": alternative.candidate_id,
                    "source_clips": [alternative],
                }
            )
        )
        changed = True
    return updated, changed


def _extract_render_qa_frames(
    rendered_video: Path,
    decisions: list[EditDecision],
    output_dir: Path,
) -> tuple[dict[str, list[Path]], float, set[str]]:
    paths: dict[str, list[Path]] = {item.segment_id: [] for item in decisions}
    targets: list[tuple[float, str, int]] = []
    for decision in decisions:
        voice_duration = max(0.01, decision.voice_end - decision.voice_start)
        for frame_index, ratio in enumerate((0.2, 0.5, 0.8), start=1):
            timestamp = decision.voice_start + voice_duration * ratio
            targets.append((timestamp, decision.segment_id, frame_index))
    targets.sort(key=lambda item: item[0])
    if not targets:
        return paths, 0.0, set()

    # The final concat MP4 may expose an inaccurate frame count to OpenCV on
    # Windows. Decode target frames with FFmpeg in one pass instead.
    render_fps = 30.0
    frame_indices = sorted({max(0, round(timestamp * render_fps)) for timestamp, _, _ in targets})
    select_expression = "+".join(f"eq(n\\,{index})" for index in frame_indices)
    output_pattern = output_dir / "selected_%04d.jpg"
    ffmpeg = find_ffmpeg()
    completed = None
    if ffmpeg:
        completed = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(rendered_video),
                "-vf",
                f"select={select_expression},scale=960:-2",
                "-fps_mode",
                "vfr",
                str(output_pattern),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    selected_files = sorted(output_dir.glob("selected_*.jpg"))
    index_to_path = {
        frame_index: selected_files[index]
        for index, frame_index in enumerate(frame_indices)
        if index < len(selected_files)
    }
    black_votes: dict[str, int] = {item.segment_id: 0 for item in decisions}
    for timestamp, segment_id, _ in targets:
        path = index_to_path.get(max(0, round(timestamp * render_fps)))
        if path is None:
            black_votes[segment_id] += 1
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            black_votes[segment_id] += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray)) < 7.0 or float(np.std(gray)) < 2.0:
            black_votes[segment_id] += 1
        paths[segment_id].append(path)

    duration = max((timestamp for timestamp, _, _ in targets), default=0.0)
    black = {segment_id for segment_id, votes in black_votes.items() if votes >= 2}
    return paths, duration, black


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.ninerouter_api_key or "local-ninerouter",
        base_url=settings.ninerouter_api_url,
    )


def _subtitle_overlap(events: list[SubtitleEvent], start: float, end: float) -> list[SubtitleEvent]:
    return [item for item in events if item.end >= start and item.start <= end]


def _estimated_voice_duration(text: str) -> float:
    words = len(re.findall(r"\w+", text, flags=re.UNICODE))
    return round(min(7.0, max(2.0, words / 2.75 + 0.35)), 3)


def _file_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


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


def _dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = unicodedata.normalize("NFC", str(item)).strip()
        if text and text not in result:
            result.append(text)
    return result[:20]


def _safe_text(value: object, fallback: str) -> str:
    text = unicodedata.normalize("NFC", str(value or "")).strip()
    return text or fallback


def _confidence(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number > 1:
        number /= 100.0
    return min(1.0, max(0.0, number))


def _optional_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return {value for value in re.findall(r"[\w]+", normalized, flags=re.UNICODE) if len(value) > 1}


def _token_overlap(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a), 1)


def _list_overlap(required: list[str], visible: list[str], default: float) -> float:
    if not required:
        return default
    visible_text = " ".join(visible)
    return sum(_token_overlap(item, visible_text) >= 0.45 for item in required) / len(required)


def _clean_tags(value: object) -> list[str]:
    tags = _string_list(value)
    cleaned: list[str] = []
    for tag in tags:
        normalized = re.sub(r"[^\w-]+", "", tag.lower().lstrip("#"), flags=re.UNICODE)
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned[:18] or ["reviewphim", "tomtatphim", "reviewphimhay"]
