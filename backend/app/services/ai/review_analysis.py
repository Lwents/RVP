from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
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
from app.services.ai.character_names import (
    CharacterNameRegistry,
    build_character_name_registry,
    canonical_character_name,
    canonicalize_character_text,
    character_name_contract,
)
from app.services.ai.context import analyze_video_context
from app.services.media.ffmpeg import find_ffmpeg, probe_video_duration
from app.services.presets import get_processing_profile
from app.services.subtitles.timing import SubtitleEvent, parse_srt


ProgressCallback = Callable[[str, int], None]


_STORY_ROLE_ORDER = ("hook", "context", "conflict", "climax", "resolution")
_STYLE_GUIDANCE = {
    "story": (
        "Ke chuyen dien anh lien mach. Dung quan he nguyen nhan-ket qua co trong evidence de noi cac event; "
        "giu bi mat cho den dung thu tu phim va chuyen doan mem, khong nhay coc."
    ),
    "fast": (
        "Nhip nhanh, cau ngan, dong tu manh. Luoc chi tiet lap lai nhung van phai giu du nguyen nhan, "
        "xung dot va ket qua de nguoi xem khong bi mat mach truyen."
    ),
    "emotional": (
        "Giau cam xuc nhung khong suy dien noi tam. Uu tien bieu cam, lua chon, mat mat, hy vong va quan he "
        "co trong emotion/actions/dialogue cua scene_facts; giam liet ke mau sac, vat dung va chi tiet trang tri "
        "khong lam thay doi cam xuc hay mach truyen."
    ),
    "funny": (
        "Duyen hai tu phan ung va tinh huong co that trong evidence. Khong che loi thoai, khong bien nhan vat "
        "thanh tro dua va khong hy sinh logic cau chuyen de lay mot cau gay cuoi."
    ),
}
_STYLE_ENFORCEMENT = {
    "story": "Da dang tu noi chuyen doan va giai thich du mat xich truoc khi sang event tiep theo.",
    "fast": "Uu tien cau 12-20 tu, bo mo ta trang tri va vao thang hanh dong-ket qua.",
    "emotional": (
        "Trong moi cum 2-3 cau, it nhat mot cau phai lam ro cam xuc quan sat duoc, lua chon, mat mat, "
        "hy vong, moi quan he hoac muc do nguy cap co trong scene_facts. Khong liet ke mau sac/do vat trang tri "
        "neu chi tiet do khong lam thay doi hanh dong, cam xuc hay ket qua."
    ),
    "funny": "Moi cau hai phai bat nguon tu phan ung/tinh huong co evidence; cam chen loi binh bia dat.",
}
_TRANSITION_CONTRACT = (
    "Khong viet nhu danh sach shot. Trong cung mot event, uu tien mo cau truc tiep bang chu the, dia diem "
    "hoac hanh dong thay vi chen tu noi tuan tu. Mot tu noi cu the khong duoc mo dau qua 10% tong so cau "
    "va khong duoc lap qua hai cau lien tiep. Chi dung quan he nguyen nhan, doi lap hay dong thoi khi "
    "cause/consequence/evidence chung minh; khong xoay vong tu dong cac tu dong nghia."
)
_STORY_ROLE_GUIDANCE = {
    "hook": "Mo bang tinh huong khoi phat hoac cau hoi trung tam, khong spoil event ve sau.",
    "context": "Gioi thieu nhan vat, boi canh, muc tieu va cac du kien can de hieu cau chuyen.",
    "conflict": "Trinh bay bien co tang dan theo quan he nguyen nhan-ket qua da duoc xac minh.",
    "climax": "Danh phan dung luong lon cho nut that, quyet dinh va cao trao co evidence manh.",
    "resolution": "Giai quyet he qua theo dung source time va chot lai mach phim, khong them danh gia ngoai evidence.",
}

# These are discourse markers, not story evidence.  They are useful sparingly,
# but a model that starts most shot descriptions with the same marker produces
# a chronological checklist instead of a narrated review.  Match only at the
# beginning of a sentence so a legitimate phrase such as "chuyện xảy ra sau
# đó" in the middle of a sentence is not penalised.
_LEADING_TRANSITION_RE = re.compile(
    r"^\s*[\"'“”‘’(\[]*\s*(?P<marker>"
    r"ngay\s+sau\s+đó|sau\s+đó|tiếp\s+đó|tiếp\s+theo|kế\s+tiếp|kế\s+đó|"
    r"rồi|lúc\s+này|ngay\s+lúc\s+ấy|đúng\s+lúc\s+ấy|cùng\s+lúc(?:\s+ấy)?|"
    r"trong\s+khi(?:\s+ấy|\s+đó)?|ở\s+nơi\s+khác|ở\s+phía\s+bên\s+kia|"
    r"thế\s+nhưng|tuy\s+nhiên|vì\s+vậy|bởi\s+vậy|do\s+đó|cuối\s+cùng"
    r")\b",
    flags=re.IGNORECASE,
)


def _leading_narrative_transition(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    match = _LEADING_TRANSITION_RE.match(normalized)
    if match is None:
        return ""
    return " ".join(match.group("marker").casefold().split())


def _narrative_transition_issues(segments: list[NarrationSegment]) -> list[str]:
    """Return deterministic issues for list-like, repetitive narration.

    The gate intentionally accepts connector-free prose.  It only rejects a
    marker that dominates the script or appears in a long identical run; this
    keeps chronology intact without encouraging blind synonym rotation.
    """

    if len(segments) < 4:
        return []
    markers = [_leading_narrative_transition(item.narration) for item in segments]
    counts = Counter(marker for marker in markers if marker)
    allowed_per_marker = max(2, math.floor(len(segments) * 0.10))
    issues: list[str] = []
    for marker, count in counts.most_common():
        if count > allowed_per_marker:
            issues.append(
                f"Từ nối '{marker}' mở đầu {count}/{len(segments)} câu; tối đa {allowed_per_marker}."
            )

    run_marker = ""
    run_start = 0
    run_length = 0
    for index, marker in enumerate(markers):
        if marker and marker == run_marker:
            run_length += 1
        else:
            if run_marker and run_length > 2:
                issues.append(
                    f"Từ nối '{run_marker}' lặp {run_length} câu liên tiếp "
                    f"từ segment {run_start + 1}."
                )
            run_marker = marker
            run_start = index
            run_length = 1 if marker else 0
    if run_marker and run_length > 2:
        issues.append(
            f"Từ nối '{run_marker}' lặp {run_length} câu liên tiếp từ segment {run_start + 1}."
        )
    return issues


_GENERIC_SEQUENCE_TRANSITIONS = {
    "ngay sau đó",
    "sau đó",
    "tiếp đó",
    "tiếp theo",
    "kế tiếp",
    "kế đó",
    "rồi",
}

_DIALOGUE_CLAIM_RE = re.compile(
    r"\b(?:nói|kể|hỏi|trả\s+lời|giải\s+thích|tiết\s+lộ|đề\s+nghị|cảnh\s+báo|"
    r"thông\s+báo|thừa\s+nhận|hứa|ra\s+lệnh|cho\s+biết|tranh\s+cãi|gọi|quyết\s+định)\b",
    flags=re.IGNORECASE,
)


def _strip_redundant_generic_transitions(
    segments: list[NarrationSegment],
) -> list[NarrationSegment]:
    """Remove only redundant within-event prefixes without changing facts.

    A first sentence may need a short event bridge.  Subsequent sentences in
    that same event already inherit chronology from their sequence, so a
    generic "Sau đó" adds no meaning.  Stripping the prefix is a safe fallback
    when the model ignored the editorial instruction; causal/contrast markers
    are never rewritten here.
    """

    cleaned: list[NarrationSegment] = []
    previous_event_id = ""
    for segment in segments:
        narration = unicodedata.normalize("NFC", segment.narration or "").strip()
        marker = _leading_narrative_transition(narration)
        if marker in _GENERIC_SEQUENCE_TRANSITIONS and segment.event_id == previous_event_id:
            match = _LEADING_TRANSITION_RE.match(narration)
            remainder = narration[match.end() :] if match is not None else narration
            remainder = re.sub(r"^\s*[,;:–—-]+\s*", "", remainder).strip()
            if remainder:
                narration = remainder[0].upper() + remainder[1:]
        if narration != segment.narration:
            segment = segment.model_copy(
                update={
                    "narration": narration,
                    "estimated_voice_duration": _estimated_voice_duration(narration),
                }
            )
        cleaned.append(segment)
        previous_event_id = segment.event_id
    return cleaned


def _style_contract(style: str) -> str:
    """Return the same selected-style rules to both the writer and QA judge."""

    selected = style if style in _STYLE_GUIDANCE else "story"
    return f"{_STYLE_GUIDANCE[selected]} {_STYLE_ENFORCEMENT[selected]} {_TRANSITION_CONTRACT}"


@dataclass(frozen=True)
class ReviewNarrationBudget:
    target_minutes: int
    target_words: int
    min_words: int
    max_words: int
    target_segments: int
    min_segments: int
    max_segments: int


@dataclass(frozen=True)
class NarrativeStyleAssessment:
    coherence_score: float
    style_score: float
    contradictions: tuple[str, ...] = ()
    early_spoilers: tuple[str, ...] = ()
    feedback: str = ""

    @property
    def logic_passed(self) -> bool:
        return (
            # A score in the high eighties can still hide a missing bridge
            # between two plot beats (the real Doraemon validation scored 88
            # while explicitly reporting such a jump).  Do not render that
            # draft; make the writer repair it first.
            self.coherence_score >= 90.0
            and not self.contradictions
            and not self.early_spoilers
        )

    @property
    def style_passed(self) -> bool:
        return self.style_score >= 85.0

    @property
    def passed(self) -> bool:
        return self.logic_passed and self.style_passed


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
    # Subtitle translation usually creates this context first.  Source-language
    # reviews do not, so start the same whole-film character pass here and let
    # it run while OpenCV detects scenes.
    context_task = asyncio.create_task(
        analyze_video_context(video_path, events, work_dir)
    )
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
        context_task.cancel()
        raise RuntimeError("Không phát hiện được scene/keyframe hợp lệ trong video.")

    try:
        film_context = await context_task
    except Exception as exc:
        print(f"Review character context fallback: {exc}")
        film_context = {}
    character_names = build_character_name_registry(film_context)

    on_progress("AI đa phương thức phân tích hình ảnh, thoại và chữ trên màn hình", 60)
    scenes = await _analyze_scene_batches(
        scenes,
        events,
        on_progress,
        batch_size=profile.review_scene_batch_size,
        concurrency=profile.review_ai_concurrency,
        character_names=character_names,
    )
    character_names = build_character_name_registry(
        film_context,
        (name for scene in scenes for name in scene.characters),
    )
    scenes = _canonicalize_scene_character_names(scenes, character_names)
    scenes = _reconcile_character_identities(scenes)
    scenes = _canonicalize_scene_character_names(scenes, character_names)

    on_progress("Lập timeline sự kiện và kiểm tra danh tính nhân vật", 71)
    story_events = await _build_event_timeline(scenes, target_minutes, style)
    story_events = _canonicalize_story_event_names(story_events, character_names)
    if not story_events:
        raise RuntimeError("Không tạo được timeline sự kiện có bằng chứng.")

    on_progress("Viết lời review từ các sự kiện đã kiểm chứng", 76)
    metadata, segments, narrative_assessment = await _write_verified_narration(
        story_events,
        scenes,
        target_minutes,
        style,
        notes,
        character_names=character_names,
    )
    metadata = _canonicalize_review_metadata(metadata, character_names)
    segments = _canonicalize_narration_names(segments, character_names)
    segments = _normalize_segments(segments, story_events, target_minutes)
    if not segments:
        raise RuntimeError("AI không tạo được câu review gắn với event hợp lệ.")

    on_progress("Ghép từng câu với cảnh và tạo EDL", 79)
    decisions = _match_segments_to_scenes(segments, story_events, scenes)
    segments, decisions, timeline_changed = await _rescore_and_stabilize_timeline(
        segments,
        story_events,
        scenes,
        decisions,
        batch_size=profile.review_scene_batch_size,
    )
    # At most three candidates are retained per sentence. Try each one once,
    # preserving source chronology, before asking the user to review it.
    for _ in range(2):
        decisions, changed = _replace_low_pre_render_matches(decisions, segments, scenes)
        if not changed:
            break
        segments, decisions, reordered = await _rescore_and_stabilize_timeline(
            segments,
            story_events,
            scenes,
            decisions,
            batch_size=profile.review_scene_batch_size,
        )
        timeline_changed = timeline_changed or reordered
    if timeline_changed:
        narrative_assessment = await _judge_narrative_style(segments, story_events, style)
    segments, decisions, narrative_assessment = await _repair_low_visual_until_stable(
        segments,
        story_events,
        scenes,
        decisions,
        style,
        target_minutes,
        narrative_assessment,
        batch_size=profile.review_scene_batch_size,
        character_names=character_names,
    )
    segments = _canonicalize_narration_names(segments, character_names)
    decisions = _canonicalize_edit_decision_names(
        decisions,
        segments,
        character_names,
    )
    report = _quality_report(
        segments,
        story_events,
        scenes,
        decisions,
        target_minutes,
        style,
        narrative_assessment,
    )

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
    character_names: CharacterNameRegistry | None = None,
) -> list[AnalyzedScene]:
    batch_size = max(1, min(8, batch_size or settings.review_scene_batch_size))
    batches = [scenes[index : index + batch_size] for index in range(0, len(scenes), batch_size)]
    semaphore = asyncio.Semaphore(max(1, min(3, concurrency)))

    async def analyze(index: int, batch: list[AnalyzedScene]) -> tuple[int, list[AnalyzedScene]]:
        async with semaphore:
            try:
                analyzed = await _analyze_scene_batch(
                    batch,
                    transcript_events,
                    character_names=character_names,
                )
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
    *,
    character_names: CharacterNameRegistry | None = None,
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
    if character_names:
        prompt += (
            " CHARACTER_NAME_CONTRACT dưới đây chỉ khóa cách viết tên, không phải bằng chứng rằng nhân vật có mặt. "
            "Khi ảnh/thoại đủ chứng minh một nhân vật đã biết, characters và mọi câu mô tả phải chép đúng canonical_names; "
            "không dịch nghĩa tên, không dùng alias. Nếu chưa đủ bằng chứng nhận dạng thì vẫn dùng UNKNOWN. "
            f"CHARACTER_NAME_CONTRACT={json.dumps(character_name_contract(character_names), ensure_ascii=False)}"
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
        model=settings.review_ai_model,
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
        # Keep the timestamped source transcript collected before the vision
        # request. The model may return only a short paraphrase (or omit its
        # dialogue evidence entirely); replacing the original here used to
        # discard the exact subtitle needed by later narration/scene QA.
        dialogue = list(
            dict.fromkeys(
                [
                    *original.evidence.dialogue,
                    *_string_list(_dict(item.get("evidence")).get("dialogue")),
                ]
            )
        )
        subtitle = list(
            dict.fromkeys(
                [
                    *original.evidence.subtitle,
                    *_string_list(_dict(item.get("evidence")).get("subtitle")),
                ]
            )
        )
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


async def _build_event_timeline(
    scenes: list[AnalyzedScene],
    target_minutes: int = 8,
    style: str = "story",
) -> list[StoryEvent]:
    usable = [scene for scene in scenes if not scene.black_frame and not _is_credits_only_scene(scene)]
    if not usable:
        usable = scenes
    budget = _review_narration_budget(target_minutes, style)
    target_event_count = min(
        len(usable),
        max(5, min(budget.min_segments, target_minutes * 4)),
    )
    min_event_count = min(target_event_count, max(4, math.floor(target_event_count * 0.80)))
    max_event_count = min(len(usable), max(target_event_count, math.ceil(target_event_count * 1.20)))
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
        "Chỉ gộp scene liên tiếp khi chúng thật sự cùng một hành động; giữ flashback/dream đúng loại; nêu quan hệ nguyên nhân-kết quả chỉ khi có bằng chứng. "
        f"Để đủ nội dung cho review {target_minutes} phút, tạo {min_event_count}-{max_event_count} event (mục tiêu {target_event_count}); "
        "không nén cả một hồi phim thành một event chung và phải phủ đều từ đầu đến cuối source. "
        "Mỗi event phải có >=1 scene_id thật. Trả JSON {events:[{event_id,order_index,scene_ids,characters,summary,cause,"
        "consequence,confidence,verification_status}]}. verification_status chỉ là verified khi có evidence trực tiếp, còn lại unverified."
    )
    try:
        response = await _client().chat.completions.create(
            model=settings.review_ai_model,
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
    if not timeline:
        timeline = _fallback_event_timeline(usable)
    return _rebalance_event_timeline(timeline, usable, target_event_count, min_event_count, max_event_count)


def _is_credits_only_scene(scene: AnalyzedScene) -> bool:
    """Trust the credits flag only when the frame contains no plot evidence.

    Multimodal models sometimes label scenes with a channel watermark or title
    card as ``credits`` even while the plot, climax or epilogue is still moving.
    Those false positives previously removed whole sections from the review.
    """

    if not scene.credits:
        return False
    credit_markers = (
        "credit",
        "staff",
        "cast list",
        "production list",
        "end card",
        "ending card",
        "title card",
        "illustration",
        "poster",
        "teaser",
        "logo",
        "rolling text",
        "static text",
        "are shown",
        "is shown",
        "displayed",
        "scrolling names",
        "character still",
        "montage still",
        "片尾",
        "制作",
        "字幕",
    )

    def credits_like(value: str) -> bool:
        normalized = unicodedata.normalize("NFKC", value).lower().strip()
        return bool(normalized) and any(marker in normalized for marker in credit_markers)

    summary = scene.event_summary.strip()
    action_text = " ".join(scene.visible_actions)
    meaningful_summary = bool(
        summary
        and summary.upper() != "UNKNOWN"
        and not credits_like(summary)
    )
    meaningful_actions = bool(
        action_text.strip()
        and not all(credits_like(action) for action in scene.visible_actions if action.strip())
    )
    meaningful_dialogue = bool(
        scene.dialogue_summary.strip()
        and not credits_like(scene.dialogue_summary)
    )
    meaningful_visual_evidence = any(
        value.strip()
        and value.strip().upper() != "UNKNOWN"
        and not credits_like(value)
        for value in scene.evidence.visual
    )
    meaningful_dialogue_evidence = any(
        value.strip()
        and value.strip().upper() != "UNKNOWN"
        and not credits_like(value)
        for value in scene.evidence.dialogue
    )
    meaningful_subtitle_evidence = any(
        value.strip()
        and value.strip().upper() != "UNKNOWN"
        and not credits_like(value)
        for value in scene.evidence.subtitle
    )
    has_plot_evidence = bool(
        meaningful_actions
        or meaningful_dialogue
        or meaningful_summary
        or meaningful_dialogue_evidence
        or meaningful_visual_evidence
        or meaningful_subtitle_evidence
    )
    return not has_plot_evidence


def _canonicalize_scene_character_names(
    scenes: list[AnalyzedScene],
    registry: CharacterNameRegistry | None,
) -> list[AnalyzedScene]:
    if not registry:
        return scenes
    normalized: list[AnalyzedScene] = []
    for scene in scenes:
        characters = _canonical_name_list(scene.characters, registry)
        normalized.append(
            scene.model_copy(
                update={
                    "characters": characters,
                    "location": canonicalize_character_text(scene.location, registry),
                    "visible_actions": _canonical_text_list(scene.visible_actions, registry),
                    "dialogue_summary": canonicalize_character_text(scene.dialogue_summary, registry),
                    "event_summary": canonicalize_character_text(scene.event_summary, registry),
                    "important_objects": _canonical_text_list(scene.important_objects, registry),
                    "emotion": canonicalize_character_text(scene.emotion, registry),
                    "evidence": ReviewEvidence(
                        visual=_canonical_text_list(scene.evidence.visual, registry),
                        dialogue=_canonical_text_list(scene.evidence.dialogue, registry),
                        subtitle=_canonical_text_list(scene.evidence.subtitle, registry),
                    ),
                }
            )
        )
    return normalized


def _canonicalize_story_event_names(
    events: list[StoryEvent],
    registry: CharacterNameRegistry | None,
) -> list[StoryEvent]:
    if not registry:
        return events
    return [
        event.model_copy(
            update={
                "characters": _canonical_name_list(event.characters, registry),
                "summary": canonicalize_character_text(event.summary, registry),
                "cause": canonicalize_character_text(event.cause, registry),
                "consequence": canonicalize_character_text(event.consequence, registry),
            }
        )
        for event in events
    ]


def _canonicalize_narration_names(
    segments: list[NarrationSegment],
    registry: CharacterNameRegistry | None,
) -> list[NarrationSegment]:
    if not registry:
        return segments
    normalized: list[NarrationSegment] = []
    for segment in segments:
        narration = canonicalize_character_text(segment.narration, registry)
        required = segment.required_visuals
        normalized.append(
            segment.model_copy(
                update={
                    "narration": narration,
                    "required_visuals": RequiredVisuals(
                        characters=_canonical_name_list(required.characters, registry),
                        actions=_canonical_text_list(required.actions, registry),
                        objects=_canonical_text_list(required.objects, registry),
                        locations=_canonical_text_list(required.locations, registry),
                    ),
                    "forbidden_visuals": _canonical_text_list(segment.forbidden_visuals, registry),
                    "purpose": canonicalize_character_text(segment.purpose, registry),
                    "estimated_voice_duration": _estimated_voice_duration(narration),
                }
            )
        )
    return normalized


def _canonicalize_edit_decision_names(
    decisions: list[EditDecision],
    segments: list[NarrationSegment],
    registry: CharacterNameRegistry | None,
) -> list[EditDecision]:
    if not registry:
        return decisions
    narration_by_id = {segment.segment_id: segment.narration for segment in segments}

    def normalize_candidate(candidate: SceneCandidate) -> SceneCandidate:
        return candidate.model_copy(
            update={
                "match_reason": canonicalize_character_text(candidate.match_reason, registry)
            }
        )

    return [
        decision.model_copy(
            update={
                "narration": narration_by_id.get(
                    decision.segment_id,
                    canonicalize_character_text(decision.narration, registry),
                ),
                "source_clips": [normalize_candidate(item) for item in decision.source_clips],
                "alternatives": [normalize_candidate(item) for item in decision.alternatives],
            }
        )
        for decision in decisions
    ]


def _canonicalize_review_metadata(
    metadata: dict,
    registry: CharacterNameRegistry | None,
) -> dict:
    if not registry or not isinstance(metadata, dict):
        return metadata
    normalized = dict(metadata)
    for key in ("title", "hook", "summary", "thumbnail_text"):
        if key in normalized:
            normalized[key] = canonicalize_character_text(str(normalized[key] or ""), registry)
    tags = normalized.get("tags")
    if isinstance(tags, list):
        normalized["tags"] = [
            canonical_character_name(str(tag), registry)
            for tag in tags
        ]
    return normalized


def _canonical_name_list(
    values: list[str],
    registry: CharacterNameRegistry,
) -> list[str]:
    result: list[str] = []
    for value in values:
        canonical = canonical_character_name(value, registry)
        if canonical and canonical not in result:
            result.append(canonical)
    return result


def _canonical_text_list(
    values: list[str],
    registry: CharacterNameRegistry,
) -> list[str]:
    result: list[str] = []
    for value in values:
        canonical = canonicalize_character_text(value, registry)
        if canonical and canonical not in result:
            result.append(canonical)
    return result


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
        has_direct_evidence = evidence_count > 0
        status = "verified" if has_direct_evidence else "unverified"
        # Keep timeline facts as data instead of pre-writing a discourse
        # marker that the narration model will copy into every sentence.
        grounded_summary = " | ".join(
            value
            for scene in linked
            if (value := scene.event_summary.strip()) and value.upper() != "UNKNOWN"
        )
        if not grounded_summary:
            grounded_summary = " | ".join(
                value for scene in linked if (value := scene.dialogue_summary.strip())
            )
        evidence_text = " ".join(
            value
            for scene in linked
            for value in (
                scene.event_summary,
                scene.dialogue_summary,
                *scene.visible_actions,
                *scene.evidence.visual,
                *scene.evidence.dialogue,
                *scene.evidence.subtitle,
            )
            if value
        )

        def grounded_relation(field: str) -> str:
            claim = _safe_text(item.get(field), "")
            return claim if claim and _token_overlap(claim, evidence_text) >= 0.35 else ""

        accepted.append(
            StoryEvent(
                event_id="pending",
                order_index=0,
                start_time=min(scene.start_time for scene in linked),
                end_time=max(scene.end_time for scene in linked),
                scene_ids=[scene.scene_id for scene in linked],
                characters=list(dict.fromkeys(name for scene in linked for name in scene.characters)),
                # The scene facts, not the model's self-declared "verified"
                # flag, are authoritative. This prevents an unrelated visual
                # evidence item from blessing a fabricated event summary.
                summary=grounded_summary or "Sự kiện có bằng chứng trực tiếp.",
                cause=grounded_relation("cause"),
                consequence=grounded_relation("consequence"),
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


def _rebalance_event_timeline(
    timeline: list[StoryEvent],
    scenes: list[AnalyzedScene],
    target_count: int,
    min_count: int,
    max_count: int,
) -> list[StoryEvent]:
    """Keep enough chronological events for the requested review runtime.

    AI commonly over-merges a long act into one event. Split only along real
    scene boundaries and use each scene's verified facts, so added detail never
    invents plot. Conversely, merge adjacent micro-events when the response is
    too fragmented for the requested narration budget.
    """

    scene_by_id = {scene.scene_id: scene for scene in scenes}
    balanced = list(sorted(timeline, key=lambda item: (item.start_time, item.order_index)))
    target_count = max(1, min(target_count, len(scenes)))
    min_count = max(1, min(min_count, target_count))
    max_count = max(target_count, max_count)

    def child_event(parent: StoryEvent, ids: list[str], keep_cause: bool, keep_consequence: bool) -> StoryEvent:
        linked = sorted(
            (scene_by_id[value] for value in ids if value in scene_by_id),
            key=lambda scene: scene.start_time,
        )
        summaries = [
            scene.event_summary.strip()
            for scene in linked
            if scene.event_summary.strip() and scene.event_summary.strip().upper() != "UNKNOWN"
        ]
        characters = list(dict.fromkeys(name for scene in linked for name in scene.characters))
        evidence_count = sum(
            len(scene.evidence.visual) + len(scene.evidence.dialogue) + len(scene.evidence.subtitle)
            for scene in linked
        )
        return parent.model_copy(
            update={
                "start_time": min(scene.start_time for scene in linked),
                "end_time": max(scene.end_time for scene in linked),
                "scene_ids": [scene.scene_id for scene in linked],
                "characters": characters or parent.characters,
                "summary": " | ".join(summaries) or parent.summary,
                "cause": parent.cause if keep_cause else "",
                "consequence": parent.consequence if keep_consequence else "",
                "confidence": min((scene.confidence for scene in linked), default=parent.confidence),
                "evidence_count": evidence_count,
                "verification_status": "verified" if evidence_count else "unverified",
            }
        )

    # A model can return the requested number of events while still choosing
    # all of them from only the first act.  Put at least one evidence-backed
    # anchor in every chronological source bucket.  Adjacent anchors are merged
    # below when this temporarily exceeds the requested event count, so the
    # final timeline stays compact without developing large unexplained jumps.
    ordered_scenes = sorted(scenes, key=lambda item: (item.start_time, item.end_time))
    bucket_count = min(target_count, len(ordered_scenes))
    claimed = {scene_id for event in balanced for scene_id in event.scene_ids}
    for raw_bucket in np.array_split(np.asarray(ordered_scenes, dtype=object), bucket_count):
        bucket = list(raw_bucket)
        if not bucket or any(scene.scene_id in claimed for scene in bucket):
            continue
        anchor = max(
            bucket,
            key=lambda scene: (
                len(scene.evidence.visual) + len(scene.evidence.dialogue) + len(scene.evidence.subtitle),
                scene.confidence,
            ),
        )
        anchor_event = _fallback_event_timeline([anchor])[0]
        balanced.append(anchor_event)
        claimed.add(anchor.scene_id)
    balanced.sort(key=lambda item: (item.start_time, item.order_index))

    # Count alone is not enough: one AI event previously swallowed the entire
    # prologue plus the first eleven minutes, yet the timeline still contained
    # 32 events overall. Its five hook sentences could not mention the missing
    # park-building transition. Split any event that is much larger than the
    # average source bucket, while allowing the final timeline to use the
    # configured max count instead of immediately merging the pieces back.
    full_span = max(1.0, ordered_scenes[-1].end_time - ordered_scenes[0].start_time)
    scene_cap = max(2, math.ceil((len(ordered_scenes) / target_count) * 1.5))
    span_cap = max(30.0, (full_span / target_count) * 1.75)
    while len(balanced) < max_count:
        oversized: list[tuple[int, StoryEvent, list[str], float]] = []
        for index, event in enumerate(balanced):
            ids = sorted(
                (value for value in event.scene_ids if value in scene_by_id),
                key=lambda value: scene_by_id[value].start_time,
            )
            if len(ids) < 2:
                continue
            ratio = max(
                len(ids) / scene_cap,
                max(0.0, event.end_time - event.start_time) / span_cap,
            )
            if ratio > 1.0:
                oversized.append((index, event, ids, ratio))
        if not oversized:
            break
        index, event, ids, _ = max(oversized, key=lambda item: item[3])
        midpoint = max(1, len(ids) // 2)
        balanced[index : index + 1] = [
            child_event(event, ids[:midpoint], True, False),
            child_event(event, ids[midpoint:], False, True),
        ]

    while len(balanced) < target_count:
        splittable = [
            (index, event)
            for index, event in enumerate(balanced)
            if len([value for value in event.scene_ids if value in scene_by_id]) >= 2
        ]
        if not splittable:
            break
        index, event = max(splittable, key=lambda value: len(value[1].scene_ids))
        ids = [value for value in event.scene_ids if value in scene_by_id]
        midpoint = max(1, len(ids) // 2)
        left = child_event(event, ids[:midpoint], True, False)
        right = child_event(event, ids[midpoint:], False, True)
        balanced[index : index + 1] = [left, right]

    if len(balanced) < min_count:
        claimed = {scene_id for event in balanced for scene_id in event.scene_ids}
        missing = [
            scene
            for scene in scenes
            if scene.scene_id not in claimed
            and (scene.evidence.visual or scene.evidence.dialogue or scene.evidence.subtitle)
        ]
        needed = min_count - len(balanced)
        if needed and missing:
            positions = np.linspace(0, len(missing) - 1, min(needed, len(missing)), dtype=int)
            fallback_by_id = {item.scene_ids[0]: item for item in _fallback_event_timeline(missing)}
            balanced.extend(fallback_by_id[missing[int(position)].scene_id] for position in positions)
            balanced.sort(key=lambda item: (item.start_time, item.order_index))

    desired = min(max_count, max(target_count, len(balanced)))
    while len(balanced) > desired and len(balanced) > 1:
        def merge_cost(index: int) -> tuple[float, float, float]:
            left, right = balanced[index], balanced[index + 1]
            ids = set(left.scene_ids + right.scene_ids)
            duration = max(left.end_time, right.end_time) - min(left.start_time, right.start_time)
            oversized_penalty = 1.0 if len(ids) > scene_cap or duration > span_cap else 0.0
            gap = max(0.0, right.start_time - left.end_time)
            return (
                oversized_penalty,
                duration / span_cap + len(ids) / scene_cap,
                gap / full_span,
            )

        pair_index = min(
            range(len(balanced) - 1),
            key=merge_cost,
        )
        left, right = balanced[pair_index], balanced[pair_index + 1]
        ids = list(dict.fromkeys(left.scene_ids + right.scene_ids))
        parent = left.model_copy(
            update={
                "summary": f"{left.summary} | {right.summary}".strip(" |"),
                "consequence": right.consequence,
                "confidence": min(left.confidence, right.confidence),
            }
        )
        balanced[pair_index : pair_index + 2] = [child_event(parent, ids, True, True)]

    balanced.sort(key=lambda item: (item.start_time, item.end_time))
    return [
        event.model_copy(update={"event_id": f"event_{index:04d}", "order_index": index})
        for index, event in enumerate(balanced, start=1)
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


def _review_narration_budget(target_minutes: int, style: str = "story") -> ReviewNarrationBudget:
    """Translate the requested runtime into a measurable script contract.

    The measured project voice reads Vietnamese review copy at roughly 195
    words/minute (229 words rendered as 68.568 seconds).
    The old prompt described that number as a *maximum*, so a valid 8-minute
    request could still return eleven short sentences and render in one minute.
    A lower and upper bound makes the requested duration an actual requirement.
    """

    minutes = max(1, min(30, int(target_minutes)))
    # The long-form validation voice averaged about 205 words/minute.  Using
    # the old 195-word estimate left an eight-minute script on the lower edge
    # of the measured duration gate.  The synthesized audio remains the final
    # authority, but this target keeps normal generations close to the user's
    # requested runtime instead of routinely under-running it.
    target_words = minutes * 205
    words_per_atomic_segment = {
        # One render window can reliably illustrate one visible action. Keep
        # sentences short enough that the model does not join two visual beats
        # merely to hit the requested total word count.
        "story": 16,
        "fast": 14,
        "emotional": 18,
        "funny": 15,
    }.get(style, 16)
    target_segments = max(8, round(target_words / words_per_atomic_segment))
    return ReviewNarrationBudget(
        target_minutes=minutes,
        target_words=target_words,
        min_words=round(target_words * 0.90),
        # Edge TTS cadence varies by punctuation and sentence length. Permit a
        # slightly wider upper script estimate, then enforce the authoritative
        # +/-10% gate on the synthesized audio before FFmpeg is allowed to run.
        max_words=round(target_words * 1.15),
        target_segments=target_segments,
        min_segments=max(6, math.floor(target_segments * 0.80)),
        max_segments=max(10, math.ceil(target_segments * 1.20)),
    )


def review_duration_adherence(actual_seconds: float, target_minutes: int) -> tuple[float, bool]:
    """Score measured TTS/video duration against the user's requested runtime.

    Script word count is the cheap pre-render contract, but Edge TTS cadence can
    still vary with punctuation and style.  The measured media is accepted only
    inside a ten-percent window; a one-minute voice can therefore never pass an
    eight-minute request merely because video and voice are mutually in sync.
    """

    target_seconds = max(1, min(30, int(target_minutes))) * 60.0
    if actual_seconds <= 0:
        return 0.0, False
    error_ratio = abs(float(actual_seconds) - target_seconds) / target_seconds
    score = max(0.0, 100.0 - error_ratio * 250.0)
    return round(score, 2), error_ratio <= 0.10


def review_narration_tempo_factor(actual_seconds: float, target_minutes: int) -> float | None:
    """Return a safe narration-only tempo correction, never a fake long stretch.

    A near-target voice may vary because of punctuation and the chosen Edge
    voice.  Up to 15% faster or 10% slower remains intelligible; anything
    farther away means the script itself is wrong and must stay blocked.
    """

    target_seconds = max(1, min(30, int(target_minutes))) * 60.0
    if actual_seconds <= 0:
        return None
    factor = float(actual_seconds) / target_seconds
    if 0.90 <= factor <= 1.15:
        return round(factor, 6)
    return None


def _event_story_roles(events: list[StoryEvent]) -> dict[str, str]:
    """Assign every chronological event to a stable five-part story arc."""

    verified = [item for item in events if item.verification_status == "verified"]
    ordered = sorted(verified or events, key=lambda item: (item.order_index, item.start_time))
    count = len(ordered)
    roles: dict[str, str] = {}
    for index, event in enumerate(ordered):
        if index == 0:
            role = "hook"
        elif index == count - 1:
            role = "resolution"
        else:
            progress = index / max(count - 1, 1)
            if progress <= 0.25:
                role = "context"
            elif progress <= 0.65:
                role = "conflict"
            elif progress <= 0.85:
                role = "climax"
            else:
                role = "resolution"
        roles[event.event_id] = role
    return roles


def _event_segment_targets(
    events: list[StoryEvent],
    budget: ReviewNarrationBudget,
    style: str,
) -> dict[str, int]:
    """Allocate enough atomic sentences across verified events and story roles."""

    ordered = [
        item
        for item in sorted(events, key=lambda value: (value.order_index, value.start_time))
        if item.verification_status == "verified"
    ]
    if not ordered:
        return {}
    roles = _event_story_roles(ordered)
    role_weights = {
        "story": {"hook": 0.08, "context": 0.20, "conflict": 0.35, "climax": 0.25, "resolution": 0.12},
        "fast": {"hook": 0.10, "context": 0.13, "conflict": 0.37, "climax": 0.30, "resolution": 0.10},
        "emotional": {"hook": 0.07, "context": 0.22, "conflict": 0.30, "climax": 0.29, "resolution": 0.12},
        "funny": {"hook": 0.10, "context": 0.18, "conflict": 0.34, "climax": 0.24, "resolution": 0.14},
    }.get(style, {"hook": 0.08, "context": 0.20, "conflict": 0.35, "climax": 0.25, "resolution": 0.12})
    by_role = {
        role: [event for event in ordered if roles[event.event_id] == role]
        for role in _STORY_ROLE_ORDER
    }
    active_weight = sum(role_weights[role] for role, values in by_role.items() if values)
    event_weights: dict[str, float] = {}
    for role, values in by_role.items():
        if not values:
            continue
        normalized_role_weight = role_weights[role] / max(active_weight, 0.001)
        duration_total = sum(max(1.0, event.end_time - event.start_time) for event in values)
        for event in values:
            duration_share = max(1.0, event.end_time - event.start_time) / duration_total
            event_weights[event.event_id] = normalized_role_weight * duration_share

    desired = max(len(ordered), budget.target_segments)
    raw = {event.event_id: desired * event_weights[event.event_id] for event in ordered}
    targets = {event.event_id: max(1, math.floor(raw[event.event_id])) for event in ordered}
    while sum(targets.values()) < desired:
        event = max(
            ordered,
            key=lambda item: (raw[item.event_id] - targets[item.event_id], -item.order_index),
        )
        targets[event.event_id] += 1
    while sum(targets.values()) > desired:
        removable = [item for item in ordered if targets[item.event_id] > 1]
        if not removable:
            break
        event = min(
            removable,
            key=lambda item: (raw[item.event_id] - targets[item.event_id], item.order_index),
        )
        targets[event.event_id] -= 1
    return targets


def _atomic_sentence_parts(text: str) -> list[str]:
    return [
        value.strip()
        for value in re.split(r"(?<=[.!?…])\s+", text)
        if value.strip()
    ] or ([text.strip()] if text.strip() else [])


def _narration_word_count(segments: list[NarrationSegment]) -> int:
    return sum(len(re.findall(r"\w+", item.narration, flags=re.UNICODE)) for item in segments)


def _narration_meets_budget(
    segments: list[NarrationSegment],
    events: list[StoryEvent],
    budget: ReviewNarrationBudget,
) -> bool:
    atomic_count = sum(len(_atomic_sentence_parts(item.narration)) for item in segments)
    words = _narration_word_count(segments)
    roles = {item.story_role for item in segments}
    required_roles = set(_event_story_roles(events).values())
    covered_events = {item.event_id for item in segments}
    required_events = {
        item.event_id for item in events if item.verification_status == "verified"
    }
    return (
        budget.min_words <= words <= budget.max_words
        and budget.min_segments <= atomic_count <= budget.max_segments
        and required_roles.issubset(roles)
        and required_events.issubset(covered_events)
    )


def _narration_requires_dialogue_evidence(text: str) -> bool:
    return bool(_DIALOGUE_CLAIM_RE.search(unicodedata.normalize("NFC", text or "")))


def _scene_dialogue_evidence(scene: AnalyzedScene) -> list[str]:
    values = [
        scene.dialogue_summary,
        *scene.evidence.dialogue,
        *scene.evidence.subtitle,
    ]
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        if not normalized or normalized.upper() in {"UNKNOWN", "[KHÔNG CÓ]"}:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
    return cleaned


def _visual_atomicity_issues(
    segments: list[NarrationSegment],
    scenes: dict[str, AnalyzedScene] | list[AnalyzedScene],
) -> list[str]:
    """Reject narration sentences that ask one rendered clip to show many scenes.

    The review renderer displays one continuous source window per narration
    sentence.  Letting the model attach three scene ids (and the union of all
    their facts) makes a sentence impossible to illustrate directly: the
    voice may describe a coin, a city transition and a magic door while the
    selected frame can only show one of them.  Keep the contract mechanical so
    a generation retry happens before TTS and FFmpeg consume any time.
    """

    scene_by_id = scenes if isinstance(scenes, dict) else {item.scene_id: item for item in scenes}
    known_character_names = {
        name
        for scene in scene_by_id.values()
        for name in scene.characters
        if name and not name.upper().startswith("UNKNOWN")
        # Generic aliases such as "ông lão" are scene-local descriptions, not
        # canonical identities that may be imposed on every other scene.
        and (name[0].isupper() or "_" in name)
    }
    issues: list[str] = []
    previous_start = float("-inf")
    for segment in segments:
        if len(_atomic_sentence_parts(segment.narration)) != 1:
            issues.append(f"segment {segment.sequence_index}: narration must be one atomic sentence")
        candidate_ids = list(dict.fromkeys(segment.candidate_scene_ids))
        if len(candidate_ids) != 1:
            issues.append(
                f"segment {segment.sequence_index}: candidate_scene_ids must contain exactly one scene"
            )
            continue
        scene = scene_by_id.get(candidate_ids[0])
        if scene is None:
            issues.append(f"segment {segment.sequence_index}: candidate scene does not exist")
            continue
        if scene.start_time + 0.25 < previous_start:
            issues.append(f"segment {segment.sequence_index}: candidate scene moves backwards")
        previous_start = max(previous_start, scene.start_time)

        required = segment.required_visuals
        local_evidence_text = " ".join(
            scene.characters
            + scene.visible_actions
            + scene.important_objects
            + [scene.location, scene.event_summary, scene.dialogue_summary]
        ).casefold()
        if (
            _narration_requires_dialogue_evidence(segment.narration)
            and not _scene_dialogue_evidence(scene)
        ):
            issues.append(
                f"segment {segment.sequence_index}: dialogue claim lacks source subtitle/dialogue evidence"
            )
        missing_named_characters = [
            name
            for name in known_character_names
            if name.casefold() in segment.narration.casefold()
            and name not in required.characters
            # "bức tượng Doraemon" is an object claim and is valid when the
            # chosen scene action/object explicitly contains that name.
            and name.casefold() not in local_evidence_text
        ]
        if missing_named_characters:
            issues.append(
                f"segment {segment.sequence_index}: narration names characters absent from required_visuals"
            )
        if len(required.actions) != 1:
            issues.append(
                f"segment {segment.sequence_index}: required_visuals.actions must contain exactly one visual action"
            )
        allowed = {
            "characters": set(scene.characters),
            "actions": set(scene.visible_actions),
            "objects": set(scene.important_objects),
            "locations": {scene.location} if scene.location else set(),
        }
        values = {
            "characters": required.characters,
            "actions": required.actions,
            "objects": required.objects,
            "locations": required.locations,
        }
        for field, requested in values.items():
            unsupported = [value for value in requested if value not in allowed[field]]
            if unsupported:
                issues.append(
                    f"segment {segment.sequence_index}: {field} contains facts from another scene"
                )
    for previous, current in zip(segments, segments[1:]):
        previous_ids = list(dict.fromkeys(previous.candidate_scene_ids))
        current_ids = list(dict.fromkeys(current.candidate_scene_ids))
        previous_action = (
            " ".join(previous.required_visuals.actions[0].casefold().split())
            if len(previous.required_visuals.actions) == 1
            else ""
        )
        current_action = (
            " ".join(current.required_visuals.actions[0].casefold().split())
            if len(current.required_visuals.actions) == 1
            else ""
        )
        if (
            current.event_id == previous.event_id
            and len(previous_ids) == len(current_ids) == 1
            and previous_ids[0] == current_ids[0]
            and previous_action
            and previous_action == current_action
            and _token_overlap(current.narration, previous.narration) >= 0.45
        ):
            issues.append(
                f"segment {current.sequence_index}: repeats the previous scene action"
            )
    return issues


def _visual_atomicity_invalid_indices(
    segments: list[NarrationSegment],
    scenes: dict[str, AnalyzedScene] | list[AnalyzedScene],
) -> list[int]:
    """Return both local contract errors and cross-sentence rewinds.

    Calling ``_visual_atomicity_issues`` with one segment at a time resets its
    chronology cursor, which used to make a backwards candidate invisible to
    the targeted repair pass.  Keep local validation cheap, then explicitly
    walk the complete candidate timeline.
    """

    scene_by_id = scenes if isinstance(scenes, dict) else {item.scene_id: item for item in scenes}
    invalid = {
        index
        for index, segment in enumerate(segments)
        if _visual_atomicity_issues([segment], scene_by_id)
    }
    previous_start = float("-inf")
    for index, segment in enumerate(segments):
        candidate_ids = list(dict.fromkeys(segment.candidate_scene_ids))
        if len(candidate_ids) != 1:
            continue
        scene = scene_by_id.get(candidate_ids[0])
        if scene is None:
            continue
        if scene.start_time + 0.25 < previous_start:
            invalid.add(index)
        previous_start = max(previous_start, scene.start_time)
        if index == 0:
            continue
        previous = segments[index - 1]
        if (
            segment.event_id == previous.event_id
            and segment.candidate_scene_ids == previous.candidate_scene_ids
            and len(segment.required_visuals.actions) == 1
            and len(previous.required_visuals.actions) == 1
            and " ".join(segment.required_visuals.actions[0].casefold().split())
            == " ".join(previous.required_visuals.actions[0].casefold().split())
            and _token_overlap(segment.narration, previous.narration) >= 0.45
        ):
            invalid.add(index)
    return sorted(invalid)


def _scene_supports_segment_required_visuals(
    segment: NarrationSegment,
    scene: AnalyzedScene,
) -> bool:
    """Require an alternative scene to support the sentence it will display."""

    required = segment.required_visuals
    return (
        set(required.characters).issubset(scene.characters)
        and set(required.actions).issubset(scene.visible_actions)
        and set(required.objects).issubset(scene.important_objects)
        and set(required.locations).issubset({scene.location} if scene.location else set())
    )


async def _repair_visual_atomic_segments(
    segments: list[NarrationSegment],
    events: list[StoryEvent],
    scenes: list[AnalyzedScene],
    style: str,
    *,
    character_names: CharacterNameRegistry | None = None,
) -> list[NarrationSegment]:
    """Repair only invalid visual sentences instead of regenerating the script.

    Long 8-minute JSON responses commonly contain just a handful of segments
    that still combine two actions. Regenerating all ~100 sentences makes the
    model damage already-valid story beats. Small targeted batches preserve the
    narrative and total word budget while fixing the actual visual contract.
    """

    event_by_id = {item.event_id: item for item in events}
    scene_by_id = {item.scene_id: item for item in scenes}
    repaired = list(segments)
    for _ in range(2):
        invalid_indices = _visual_atomicity_invalid_indices(repaired, scene_by_id)
        if not invalid_indices:
            break
        changed = False
        for offset in range(0, len(invalid_indices), 8):
            batch_indices = invalid_indices[offset : offset + 8]
            items: list[dict] = []
            for index in batch_indices:
                segment = repaired[index]
                event = event_by_id.get(segment.event_id)
                if event is None:
                    continue
                previous_start = next(
                    (
                        scene_by_id[previous.candidate_scene_ids[0]].start_time
                        for previous in reversed(repaired[:index])
                        if len(previous.candidate_scene_ids) == 1
                        and previous.candidate_scene_ids[0] in scene_by_id
                    ),
                    float("-inf"),
                )
                next_start = next(
                    (
                        scene_by_id[following.candidate_scene_ids[0]].start_time
                        for following in repaired[index + 1 :]
                        if len(following.candidate_scene_ids) == 1
                        and following.candidate_scene_ids[0] in scene_by_id
                    ),
                    float("inf"),
                )
                linked = [
                    scene_by_id[value]
                    for value in event.scene_ids
                    if value in scene_by_id
                    and scene_by_id[value].start_time + 0.25 >= previous_start
                    and scene_by_id[value].start_time <= next_start + 0.25
                ]
                word_count = len(re.findall(r"\w+", segment.narration, flags=re.UNICODE))
                items.append(
                    {
                        "sequence_index": segment.sequence_index,
                        "event_id": segment.event_id,
                        "story_role": segment.story_role,
                        "selected_style": style,
                        "original_narration": segment.narration,
                        "target_words": word_count,
                        "previous_narration": repaired[index - 1].narration if index > 0 else "",
                        "next_narration": repaired[index + 1].narration if index + 1 < len(repaired) else "",
                        "previous_action": (
                            repaired[index - 1].required_visuals.actions[0]
                            if index > 0 and len(repaired[index - 1].required_visuals.actions) == 1
                            else ""
                        ),
                        "next_action": (
                            repaired[index + 1].required_visuals.actions[0]
                            if index + 1 < len(repaired)
                            and len(repaired[index + 1].required_visuals.actions) == 1
                            else ""
                        ),
                        "min_source_time": previous_start if math.isfinite(previous_start) else None,
                        "max_source_time": next_start if math.isfinite(next_start) else None,
                        "scene_facts": [
                            {
                                "scene_id": scene.scene_id,
                                "start_time": scene.start_time,
                                "characters": scene.characters,
                                "actions": scene.visible_actions,
                                "objects": scene.important_objects,
                                "location": scene.location,
                                "emotion": scene.emotion,
                                "dialogue": scene.dialogue_summary,
                                "source_dialogue_subtitle": _scene_dialogue_evidence(scene),
                            }
                            for scene in linked
                        ],
                    }
                )
            if not items:
                continue
            prompt = (
                "Sua cuc bo cac cau review, khong viet lai phan khac. Tra JSON {segments:[...]}. "
                "Moi output giu nguyen sequence_index va event_id, co narration, required_visuals "
                "{characters,actions,objects,locations}, candidate_scene_ids, confidence, purpose. "
                "candidate_scene_ids phai co dung 1 scene_id; required_visuals.actions phai co dung 1 action "
                "copy nguyen van tu scene do; moi required fact khac cung phai copy tu scene do. Narration chi "
                "mo ta action trung tam nay, la mot cau, khong ke them action cu trong menh de sau-khi/truoc-khi/"
                "trong-khi. Uu tien mo truc tiep bang chu the/dia diem/hanh dong; khong them 'Sau do' hay xoay vong "
                "tu dong nghia. Cam dung 'va/nhung/roi' de noi them chu the hoac hanh dong thu hai. "
                "Chi dung quan he chuyen tiep neu scene_facts chung minh. So tu moi cau nam "
                "trong target_words +/-3, giu dung selected_style, khong them chi tiet ngoai scene_facts. "
                "Neu narration ke noi dung noi/hoi/giai thich/tiet lo/de nghi/canh bao/tranh cai/quyet dinh thi "
                "dialogue hoac source_dialogue_subtitle cua CHINH scene do phai truc tiep xac nhan y nghia; "
                "neu khong co sub/thoai phu hop thi viet lai thanh hanh dong nhin thay, cam suy dien loi noi. "
                "Khong duoc lap lai cung scene/action cua previous_action hoac next_action; neu cau cu bi lap, "
                "phai chon mot action khac co that trong scene_facts va van nam dung thu tu source."
            )
            if character_names:
                prompt += (
                    " Moi ten nhan vat phai chep dung CHARACTER_NAME_CONTRACT, khong dich ten va khong dung alias: "
                    + json.dumps(character_name_contract(character_names), ensure_ascii=False)
                )
            try:
                response = await _client().chat.completions.create(
                    model=settings.review_ai_model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    timeout=180,
                )
                data = _loads_json(response.choices[0].message.content or "{}")
                values = data.get("segments", []) if isinstance(data, dict) else []
            except Exception as exc:
                print(f"Visual-atomic narration repair fallback: {exc}")
                continue
            if not isinstance(values, list):
                continue
            original_by_sequence = {
                repaired[index].sequence_index: (index, repaired[index])
                for index in batch_indices
            }
            for item in values:
                if not isinstance(item, dict):
                    continue
                try:
                    sequence_index = int(item.get("sequence_index"))
                except (TypeError, ValueError):
                    continue
                original_value = original_by_sequence.get(sequence_index)
                if original_value is None:
                    continue
                index, original = original_value
                if str(item.get("event_id") or "") != original.event_id:
                    continue
                parsed = _segments_from_narration_payload({"segments": [item]}, events)
                if len(parsed) != 1:
                    continue
                candidate = parsed[0].model_copy(
                    update={
                        "segment_id": original.segment_id,
                        "sequence_index": original.sequence_index,
                        "story_role": original.story_role,
                    }
                )
                candidate = _canonicalize_narration_names(
                    [candidate],
                    character_names,
                )[0]
                original_words = len(re.findall(r"\w+", original.narration, flags=re.UNICODE))
                candidate_words = len(re.findall(r"\w+", candidate.narration, flags=re.UNICODE))
                if abs(candidate_words - original_words) > 3:
                    continue
                if _visual_atomicity_issues([candidate], scene_by_id):
                    continue
                candidate_scene = scene_by_id[candidate.candidate_scene_ids[0]]
                previous_start = next(
                    (
                        scene_by_id[previous.candidate_scene_ids[0]].start_time
                        for previous in reversed(repaired[:index])
                        if len(previous.candidate_scene_ids) == 1
                        and previous.candidate_scene_ids[0] in scene_by_id
                    ),
                    float("-inf"),
                )
                next_start = next(
                    (
                        scene_by_id[following.candidate_scene_ids[0]].start_time
                        for following in repaired[index + 1 :]
                        if len(following.candidate_scene_ids) == 1
                        and following.candidate_scene_ids[0] in scene_by_id
                    ),
                    float("inf"),
                )
                if (
                    candidate_scene.start_time + 0.25 < previous_start
                    or candidate_scene.start_time > next_start + 0.25
                ):
                    continue
                repaired[index] = candidate
                changed = True
        if not changed:
            break
    return repaired


def _segments_from_narration_payload(
    data: dict,
    events: list[StoryEvent],
) -> list[NarrationSegment]:
    raw_segments = data.get("segments", [])
    valid_events = {event.event_id: event for event in events}
    story_roles = _event_story_roles(events)
    segments: list[NarrationSegment] = []
    if isinstance(raw_segments, list):
        for sequence_index, item in enumerate(raw_segments, start=1):
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
                    story_role=story_roles.get(event_id, "context"),
                    sequence_index=sequence_index,
                )
            )
    return segments


def _assessment_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        text
        for item in value
        if (text := _safe_text(item, "").strip())
    )


def _parse_narrative_style_assessment(data: object) -> NarrativeStyleAssessment:
    payload = data if isinstance(data, dict) else {}
    contradictions = _assessment_strings(payload.get("contradictions"))
    early_spoilers = _assessment_strings(payload.get("early_spoilers"))
    coherence = _confidence(payload.get("coherence_score")) * 100.0
    style_score = _confidence(payload.get("style_score")) * 100.0
    # The generic confidence parser treats 0..1 as its native range. Judges
    # commonly return 0..100, so preserve those values before clamping.
    for key, current in (("coherence_score", coherence), ("style_score", style_score)):
        raw = payload.get(key)
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            numeric = current
        if numeric > 1.0:
            numeric = max(0.0, min(100.0, numeric))
        else:
            numeric = max(0.0, min(1.0, numeric)) * 100.0
        if key == "coherence_score":
            coherence = numeric
        else:
            style_score = numeric
    if contradictions or early_spoilers:
        coherence = min(coherence, 84.0)
    return NarrativeStyleAssessment(
        coherence_score=round(coherence, 2),
        style_score=round(style_score, 2),
        contradictions=contradictions,
        early_spoilers=early_spoilers,
        feedback=_safe_text(payload.get("feedback"), ""),
    )


async def _judge_narrative_style(
    segments: list[NarrationSegment],
    events: list[StoryEvent],
    style: str,
) -> NarrativeStyleAssessment:
    event_by_id = {event.event_id: event for event in events}
    judge_input = {
        "selected_style": style,
        "style_contract": _style_contract(style),
        "events": [
            {
                "event_id": event.event_id,
                "order_index": event.order_index,
                "start_time": event.start_time,
                "summary": event.summary,
                "cause": event.cause,
                "consequence": event.consequence,
            }
            for event in events
            if event.verification_status == "verified"
        ],
        "narration": [
            {
                "sequence_index": segment.sequence_index,
                "event_id": segment.event_id,
                "story_role": segment.story_role,
                "event_order": event_by_id[segment.event_id].order_index,
                "text": segment.narration,
                # These values are copied from the linked scene facts during
                # generation. Giving them to the independent judge prevents a
                # real on-screen prop/action from being misclassified as a
                # hallucination merely because the compact event summary did
                # not mention every visible detail.
                "required_visuals": segment.required_visuals.model_dump(),
                "candidate_scene_ids": segment.candidate_scene_ids,
            }
            for segment in segments
            if segment.event_id in event_by_id
        ],
    }
    prompt = (
        "Bạn là QA biên tập review phim. Chỉ dùng EVENTS để chấm NARRATION. "
        "coherence_score 0..100 đánh giá: chuyển đoạn liền mạch, quan hệ nguyên nhân-kết quả có bằng chứng, "
        "không mâu thuẫn nhân vật/hành động, không nhảy mất mắt xích và hook không spoil sớm cao trào/kết thúc. "
        "style_score 0..100 đánh giá lời kể có thể hiện đúng selected_style/style_contract hay không mà không bịa; "
        "phải trừ mạnh nếu câu chữ giống danh sách mô tả shot hoặc lặp cùng một từ nối. "
        "Liệt kê contradictions và early_spoilers cụ thể; feedback ngắn, có thể dùng để viết lại. "
        "Trả JSON {coherence_score,style_score,contradictions:[],early_spoilers:[],feedback}."
    )
    try:
        response = await _client().chat.completions.create(
            model=settings.review_ai_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(judge_input, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=180,
        )
        assessment = _parse_narrative_style_assessment(
            _loads_json(response.choices[0].message.content or "{}")
        )
        transition_issues = _narrative_transition_issues(segments)
        if transition_issues:
            detail = " ".join(transition_issues[:3])
            return NarrativeStyleAssessment(
                coherence_score=assessment.coherence_score,
                style_score=min(84.0, assessment.style_score),
                contradictions=assessment.contradictions,
                early_spoilers=assessment.early_spoilers,
                feedback=f"{detail} {assessment.feedback}".strip(),
            )
        return assessment
    except Exception as exc:
        return NarrativeStyleAssessment(
            coherence_score=0.0,
            style_score=0.0,
            feedback=f"Không chạy được AI narrative/style QA: {exc}",
        )


async def _write_verified_narration(
    events: list[StoryEvent],
    scenes: list[AnalyzedScene],
    target_minutes: int,
    style: str,
    notes: str | None,
    *,
    character_names: CharacterNameRegistry | None = None,
) -> tuple[dict, list[NarrationSegment], NarrativeStyleAssessment]:
    scene_by_id = {scene.scene_id: scene for scene in scenes}
    budget = _review_narration_budget(target_minutes, style)
    story_roles = _event_story_roles(events)
    event_targets = _event_segment_targets(events, budget, style)
    ordered_events = sorted(events, key=lambda item: (item.order_index, item.start_time))
    event_data = []
    for index, event in enumerate(ordered_events):
        linked = [scene_by_id[value] for value in event.scene_ids if value in scene_by_id]
        event_data.append(
            {
                **event.model_dump(),
                "story_role": story_roles.get(event.event_id, "context"),
                "requested_segments": event_targets.get(event.event_id, 0),
                "previous_event_summary": ordered_events[index - 1].summary if index > 0 else "",
                "next_event_summary": ordered_events[index + 1].summary if index + 1 < len(ordered_events) else "",
                "scene_facts": [
                    {
                        "scene_id": scene.scene_id,
                        "characters": scene.characters,
                        "actions": scene.visible_actions,
                        "objects": scene.important_objects,
                        "location": scene.location,
                        "emotion": scene.emotion,
                        "dialogue": scene.dialogue_summary,
                        "summary": scene.event_summary,
                        "evidence": scene.evidence.model_dump(),
                    }
                    for scene in linked
                ],
            }
        )

    arc_contract = " ".join(
        f"{role.upper()}: {_STORY_ROLE_GUIDANCE[role]}"
        for role in _STORY_ROLE_ORDER
    )
    style_enforcement = _STYLE_ENFORCEMENT.get(style, _STYLE_ENFORCEMENT["story"])
    average_words = max(10, round(budget.target_words / max(budget.target_segments, 1)))
    prompt = (
        "Ban la bien tap vien review phim. Viet LOI DOC TIENG VIET chi tu EVENT TIMELINE da kiem chung; "
        "cam dung kien thuc ngoai du lieu, doi ten, doi nguoi hanh dong, dao source time hoac spoil event chua toi. "
        f"Thoi luong bat buoc {budget.target_minutes} phut: viet {budget.min_words}-{budget.max_words} tu "
        f"(muc tieu {budget.target_words}), khong duoc coi day la gioi han toi da. Tao {budget.min_segments}-"
        f"{budget.max_segments} segments (muc tieu {budget.target_segments}) theo requested_segments cua tung event. "
        f"Muc tieu moi cau {max(8, average_words - 2)}-{average_words + 2} tu; khong keo dai bang tinh tu, mau sac "
        "hay vat dung trang tri neu chung khong lam thay doi hanh dong/cam xuc/ket qua. Tong so tu quan trong hon "
        "viec viet moi cau dai. "
        "Mot event co the co nhieu segments lien tiep de dat dung thoi luong, nhung moi segment chi duoc la MOT cau atomic "
        "voi MOT y chinh co evidence; khong ghep hai su kien vao mot cau. Moi segment phai gan event_id that, di theo "
        "source time tang dan va co story_role dung nhu event. Cau dau event moi BAT BUOC tom dung hanh dong trung tam "
        "trong summary cua event do. Neu cause/consequence/evidence chung minh quan he voi event truoc, dien dat quan "
        "he do ngan gon ma khong them hanh dong thu hai; neu khong, mo thang bang chu the, dia diem hoac hanh dong, "
        "khong chen tu noi chung chung. "
        f"Cau truc bat buoc theo thu tu: {arc_contract} "
        f"Phong cach da chon ({style}): {_style_contract(style)} "
        "Bo event unverified/UNKNOWN. MOI SEGMENT CHI DUOC MINH HOA BOI MOT CANH: candidate_scene_ids bat buoc "
        "la list co DUNG 1 scene_id thuoc event, va narration chi duoc mo ta nguoi/hang dong/vat the/boi canh co "
        "trong scene_facts cua scene_id duy nhat do. Cam gop chuyen canh, cam ke hanh dong tu scene khac trong cung "
        "mot segment. required_visuals.actions bat buoc co DUNG 1 action copy nguyen van tu scene do; narration "
        "chi xoay quanh action trung tam nay. Neu narration ke noi dung noi/hoi/giai thich/tiet lo/de nghi/canh bao/"
        "tranh cai/quyet dinh thi dialogue hoac evidence.dialogue/subtitle cua CHINH scene do phai truc tiep xac "
        "nhan; neu khong co sub/thoai phu hop thi viet lai thanh hanh dong nhin thay, cam suy dien noi dung loi noi. "
        "Cam dung 'va/nhung/roi' de noi them chu the hoac hanh dong thu hai. "
        "Chi segment DAU TIEN cua event moi duoc lam cau chuyen doan. Ben trong "
        "cung event, mo cau truc tiep bang chu the/dia diem/hanh dong; khong lap 'Sau do', khong thay bang mot vong "
        "tu dong nghia. Tu noi nguyen nhan, doi lap, dong thoi chi duoc dung khi evidence chung minh va khong duoc "
        "ke them action khac trong menh de 'sau khi X', 'truoc khi X', 'trong khi X'. "
        "Cac required_visuals khac phai cu the va moi gia tri phai copy nguyen van tu scene_facts cua scene_id "
        "duy nhat; cac candidate scene cua cac segment phai tang dan theo source time. "
        "Tra mot JSON object gom title,hook,summary,thumbnail_text,tags,segments. Moi segment gom event_id,narration,"
        "story_role,required_visuals{characters,actions,objects,locations},forbidden_visuals,candidate_scene_ids,"
        "confidence,purpose. Hook khong duoc tiet lo cao trao hoac ket phim. Summary phai tom tat dung thu tu cau chuyen."
    )
    if character_names:
        prompt += (
            " CHARACTER_NAME_CONTRACT la bat buoc cho title, hook, summary, narration, purpose va required_visuals: "
            "chi dung dung canonical_names, khong dich nghia ten rieng, khong viet alias. "
            f"CHARACTER_NAME_CONTRACT={json.dumps(character_name_contract(character_names), ensure_ascii=False)}"
        )
    if notes:
        prompt += f" Ghi chu nguoi dung, chi ap dung neu khong mau thuan evidence: {notes[:1500]}"

    best_segments: list[NarrationSegment] = []
    best_candidate_data: dict = {}
    best_distance = float("inf")
    best_valid: tuple[dict, list[NarrationSegment], NarrativeStyleAssessment] | None = None
    best_judge_score = float("-inf")
    correction = ""
    last_error: Exception | None = None
    last_words = 0
    last_atomic_count = 0
    last_visual_issues: list[str] = []
    last_transition_issues: list[str] = []
    for attempt in range(4):
        try:
            response = await _client().chat.completions.create(
                model=settings.review_ai_model,
                messages=[
                    {"role": "system", "content": prompt + correction},
                    {"role": "user", "content": json.dumps(event_data, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                temperature=0.2 if attempt == 0 else 0.1,
                timeout=240,
            )
            candidate_data = _loads_json(response.choices[0].message.content or "{}")
            if not isinstance(candidate_data, dict):
                candidate_data = {}
            candidate_segments = _segments_from_narration_payload(candidate_data, events)
            candidate_segments = _canonicalize_narration_names(
                candidate_segments,
                character_names,
            )
            candidate_segments = await _repair_visual_atomic_segments(
                candidate_segments,
                events,
                scenes,
                style,
                character_names=character_names,
            )
            candidate_segments = _canonicalize_narration_names(
                candidate_segments,
                character_names,
            )
            candidate_segments = _strip_redundant_generic_transitions(candidate_segments)
            words = _narration_word_count(candidate_segments)
            atomic_count = sum(len(_atomic_sentence_parts(item.narration)) for item in candidate_segments)
            visual_issues = _visual_atomicity_issues(candidate_segments, scene_by_id)
            transition_issues = _narrative_transition_issues(candidate_segments)
            last_words = words
            last_atomic_count = atomic_count
            last_visual_issues = visual_issues
            last_transition_issues = transition_issues
            distance = abs(words - budget.target_words) / max(budget.target_words, 1)
            distance += abs(atomic_count - budget.target_segments) / max(budget.target_segments, 1)
            distance += min(1.0, len(transition_issues) * 0.25)
            if distance < best_distance:
                best_segments = candidate_segments
                best_candidate_data = candidate_data
                best_distance = distance
            if (
                _narration_meets_budget(candidate_segments, events, budget)
                and not visual_issues
                and not transition_issues
            ):
                assessment = await _judge_narrative_style(candidate_segments, events, style)
                judge_score = assessment.coherence_score + assessment.style_score
                if judge_score > best_judge_score:
                    best_valid = (candidate_data, candidate_segments, assessment)
                    best_judge_score = judge_score
                if assessment.passed:
                    return candidate_data, candidate_segments, assessment
                judge_details = list(assessment.contradictions) + list(assessment.early_spoilers)
                judge_feedback = "; ".join(judge_details[:4]) or assessment.feedback
                correction = (
                    " LAN TRUOC DU THOI LUONG NHUNG QA KHONG DAT. VIET LAI TOAN BO JSON, "
                    f"sua dung feedback nay: {judge_feedback[:1200]}. "
                    f"Coherence can >=90 (dang {assessment.coherence_score:.1f}), style {style} can >=85 "
                    f"(dang {assessment.style_score:.1f}); khong duoc doi event/source order. "
                    "Moi event phai co cau dau bao quat summary, khong thay no bang mot chi tiet phu. "
                    f"Ap dung nghiem quy tac phong cach: {style_enforcement}"
                )
            else:
                corrections: list[str] = []
                if not _narration_meets_budget(candidate_segments, events, budget):
                    corrections.append(
                        f"so luong dang la {words} tu/{atomic_count} cau; bat buoc nam trong "
                        f"{budget.min_words}-{budget.max_words} tu va {budget.min_segments}-{budget.max_segments} cau"
                    )
                if visual_issues:
                    corrections.append(
                        "moi segment chi co DUNG MOT candidate_scene_id va DUNG MOT required_visuals.actions; "
                        "narration chi mo ta action trung tam cua scene do, required_visuals chi copy facts cua "
                        "chinh scene do, va khong them action cu trong menh de 'sau khi', 'truoc khi', 'trong khi'; "
                        "neu ke noi dung loi noi thi subtitle/dialogue cua chinh scene phai xac nhan truc tiep; "
                        "chi dung chuyen tiep trung tinh ngan; loi mau: "
                        + "; ".join(visual_issues[:8])
                    )
                if transition_issues:
                    corrections.append(
                        "loi doc dang thanh danh sach shot. Trong cung event, bo tu noi tuan tu va mo bang chu the/"
                        "dia diem/hanh dong; chi dung quan he co evidence, khong xoay vong tu dong nghia. Loi: "
                        + "; ".join(transition_issues[:6])
                    )
                correction = (
                    " LAN TRUOC KHONG DAT. Hay VIET LAI TOAN BO JSON VA SUA DONG THOI TAT CA LOI SAU: "
                    + " | ".join(corrections)
                    + ". Giu du requested_segments va moi rang buoc da dat."
                )
        except Exception as exc:
            last_error = exc
            correction = " LAN TRUOC LOI HOAC JSON KHONG HOP LE. Hay tao lai day du dung hop dong tren."

    if best_valid is not None:
        # Keep the best duration-valid draft visible for manual QA. The quality
        # report below blocks rendering when the narrative judge is <90 or the
        # selected-style score is <85.
        return best_valid

    if best_segments:
        # A nearly valid draft is still useful: the editor can deterministically
        # repair its timeline/candidate choices, or show a preview with warnings.
        # Failing the whole job here previously forced the user to re-run a long
        # multimodal analysis just because one-sentence/one-scene validation was
        # overly strict.
        try:
            assessment = await _judge_narrative_style(best_segments, events, style)
        except Exception:
            assessment = NarrativeStyleAssessment(
                coherence_score=0.0,
                style_score=0.0,
                feedback="Không chạy được narrative QA cho bản nháp gần nhất.",
            )
        return best_candidate_data, best_segments, assessment

    words = _narration_word_count(best_segments)
    atomic_count = sum(len(_atomic_sentence_parts(item.narration)) for item in best_segments)
    detail = f" AI error: {last_error}" if last_error and not best_segments else ""
    constraint_detail = (
        f" Lan cuoi: {last_words} tu/{last_atomic_count} cau, "
        f"{len(last_visual_issues)} loi mot-cau-mot-canh, "
        f"{len(last_transition_issues)} loi lap tu noi."
        if last_words or last_atomic_count or last_visual_issues or last_transition_issues
        else ""
    )
    raise RuntimeError(
        "AI khong tao du kich ban cho thoi luong da chon: "
        f"can {budget.min_words}-{budget.max_words} tu/{budget.min_segments}-{budget.max_segments} cau, "
        f"ban gan muc tieu nhat {words} tu/{atomic_count} cau.{constraint_detail}{detail}"
    )


def _normalize_segments(
    segments: list[NarrationSegment],
    events: list[StoryEvent],
    target_minutes: int,
) -> list[NarrationSegment]:
    order = {event.event_id: event.order_index for event in events}
    story_roles = _event_story_roles(events)
    normalized: list[NarrationSegment] = []
    for segment in sorted(segments, key=lambda item: order.get(item.event_id, 999999)):
        # The inspector edits one narration sentence at a time. Split multi-sentence
        # responses without changing their evidence/event binding.
        sentences = _atomic_sentence_parts(segment.narration)
        for sentence in sentences:
            normalized.append(
                segment.model_copy(
                    update={
                        "segment_id": f"segment_{len(normalized) + 1:04d}",
                        "narration": sentence,
                        "estimated_voice_duration": _estimated_voice_duration(sentence),
                        "story_role": story_roles.get(segment.event_id, "context"),
                        "sequence_index": len(normalized) + 1,
                    }
                )
            )
    max_segments = max(12, target_minutes * 15)
    return normalized[:max_segments]


def _reorder_intra_event_segments_by_verified_time(
    segments: list[NarrationSegment],
    decisions: list[EditDecision],
) -> tuple[list[NarrationSegment], bool]:
    """Order visual beats inside one event by their verified keyframe time.

    Two different actions can share one broad detected scene.  Their scene
    start is identical, so the pre-AI chronology check cannot distinguish
    them.  Multimodal QA supplies the real keyframe timestamp; use it to order
    only contiguous sentences from the same event, never to move one story
    event across another.
    """

    source_start = {
        decision.segment_id: decision.source_clips[0].start_seconds
        for decision in decisions
        if decision.source_clips
    }
    ordered: list[NarrationSegment] = []
    changed = False
    index = 0
    while index < len(segments):
        end = index + 1
        while end < len(segments) and segments[end].event_id == segments[index].event_id:
            end += 1
        group = segments[index:end]
        if len(group) > 1 and all(item.segment_id in source_start for item in group):
            sorted_group = sorted(
                group,
                key=lambda item: (source_start[item.segment_id], item.sequence_index),
            )
            changed = changed or [item.segment_id for item in sorted_group] != [
                item.segment_id for item in group
            ]
            ordered.extend(sorted_group)
        else:
            ordered.extend(group)
        index = end

    if not changed:
        return segments, False
    return [
        segment.model_copy(
            update={
                "sequence_index": sequence_index,
            }
        )
        for sequence_index, segment in enumerate(ordered, start=1)
    ], True


def _reorder_decisions_to_segment_order(
    segments: list[NarrationSegment],
    decisions: list[EditDecision],
) -> list[EditDecision]:
    """Keep verified clip scores while rebuilding only the voice order."""

    decision_by_id = {item.segment_id: item for item in decisions}
    ordered: list[EditDecision] = []
    cursor = 0.0
    for segment in segments:
        decision = decision_by_id.get(segment.segment_id)
        if decision is None:
            continue
        duration = max(0.01, decision.voice_end - decision.voice_start)
        ordered.append(
            decision.model_copy(
                update={
                    "narration": segment.narration,
                    "voice_start": round(cursor, 3),
                    "voice_end": round(cursor + duration, 3),
                }
            )
        )
        cursor += duration
    return ordered


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
    last_source_start = float("-inf")
    event_segment_totals: dict[str, int] = {}
    event_segment_positions: dict[str, int] = {}
    for item in segments:
        event_segment_totals[item.event_id] = event_segment_totals.get(item.event_id, 0) + 1
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
        chronological_candidates = [
            candidate
            for candidate in candidates
            if candidate.start_seconds + 0.25 >= last_source_start
        ]
        candidate_pool = chronological_candidates or candidates
        best_score = max(candidate.match_score for candidate in candidate_pool)
        close_matches = [
            candidate
            for candidate in candidate_pool
            if candidate.match_score >= best_score - 0.06
        ]
        unused_close_matches = [
            candidate for candidate in close_matches if candidate.scene_id not in recent[-2:]
        ]
        selection_pool = unused_close_matches or close_matches

        # A generation that passed the visual-atomic contract has one explicit
        # source scene for this sentence.  Honour it instead of letting the
        # event-level spreading heuristic silently replace it with a different
        # visual beat.  Multi-candidate/manual legacy data keeps the spreading
        # behaviour below.
        explicit = None
        explicit_ids = list(dict.fromkeys(segment.candidate_scene_ids))
        if len(explicit_ids) == 1:
            explicit = next(
                (
                    candidate
                    for candidate in candidate_pool
                    if candidate.scene_id == explicit_ids[0]
                    and candidate.start_seconds + 0.25 >= last_source_start
                ),
                None,
            )

        # Spread the narration of one event across that event's real source
        # scenes in chronological order. Previously the highest semantic score
        # could select a late scene on the first sentence; the global no-rewind
        # rule then forced every remaining sentence to repeat that late scene,
        # producing a logically ordered script whose video covered only a small
        # part of the movie.
        chronological_scene_ids = sorted(
            {candidate.scene_id for candidate in candidate_pool},
            key=lambda scene_id: (scene_by_id[scene_id].start_time, scene_by_id[scene_id].end_time),
        )
        scene_rank = {scene_id: index for index, scene_id in enumerate(chronological_scene_ids)}
        event_position = event_segment_positions.get(segment.event_id, 0)
        event_total = event_segment_totals.get(segment.event_id, 1)
        event_segment_positions[segment.event_id] = event_position + 1
        if event_total <= 1:
            ideal_rank = (len(chronological_scene_ids) - 1) / 2.0
        else:
            ideal_rank = event_position * (len(chronological_scene_ids) - 1) / (event_total - 1)
        selected = explicit or min(
            selection_pool,
            key=lambda item: (
                abs(scene_rank[item.scene_id] - ideal_rank),
                -item.match_score,
                item.start_seconds,
            ),
        )
        recent.append(selected.scene_id)
        last_source_start = max(last_source_start, selected.start_seconds)
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
                alternatives=[selected]
                + [item for item in candidates if item.candidate_id != selected.candidate_id][:2],
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


def _review_source_coverage_score(
    scenes: list[AnalyzedScene],
    decisions: list[EditDecision],
    target_minutes: int,
) -> float:
    """Measure selected clips against evidence across the whole source movie."""

    evidence_scenes = sorted(
        (
            scene
            for scene in scenes
            if not scene.black_frame
            and not _is_credits_only_scene(scene)
            and (scene.evidence.visual or scene.evidence.dialogue or scene.evidence.subtitle)
        ),
        key=lambda item: (item.start_time, item.end_time),
    )
    if not evidence_scenes:
        return 0.0

    selected_ids = {
        decision.source_clips[0].scene_id
        for decision in decisions
        if decision.source_clips
    }
    selected_scenes = [scene for scene in evidence_scenes if scene.scene_id in selected_ids]
    if not selected_scenes:
        return 0.0

    bucket_count = min(
        len(evidence_scenes),
        max(4, min(max(1, int(target_minutes)) * 4, len(evidence_scenes))),
    )
    raw_buckets = np.array_split(np.asarray(evidence_scenes, dtype=object), bucket_count)
    covered_buckets = sum(
        any(scene.scene_id in selected_ids for scene in list(raw_bucket))
        for raw_bucket in raw_buckets
    )
    bucket_coverage = covered_buckets / max(bucket_count, 1)

    full_start = evidence_scenes[0].start_time
    full_end = max(scene.end_time for scene in evidence_scenes)
    covered_start = min(scene.start_time for scene in selected_scenes)
    covered_end = max(scene.end_time for scene in selected_scenes)
    full_span = max(0.0, full_end - full_start)
    span_coverage = (
        1.0
        if full_span <= 0.001
        else min(1.0, max(0.0, (covered_end - covered_start) / full_span))
    )
    return round(100.0 * (bucket_coverage * 0.65 + span_coverage * 0.35), 2)


def _quality_report(
    segments: list[NarrationSegment],
    events: list[StoryEvent],
    scenes: list[AnalyzedScene],
    decisions: list[EditDecision],
    target_minutes: int = 8,
    style: str = "story",
    narrative_assessment: NarrativeStyleAssessment | None = None,
) -> ReviewQualityReport:
    scores = [item.source_clips[0].match_score for item in decisions if item.source_clips]
    direct = 100.0 * sum(score >= 0.75 for score in scores) / max(len(segments), 1)
    event_order = {event.event_id: event.order_index for event in events}
    order_values = [event_order.get(item.event_id, 999999) for item in decisions]
    event_order_ok = all(a <= b for a, b in zip(order_values, order_values[1:]))
    source_starts = [item.source_clips[0].start_seconds for item in decisions if item.source_clips]
    source_order_ok = all(left <= right + 0.25 for left, right in zip(source_starts, source_starts[1:]))
    chronology = 100.0 if event_order_ok and source_order_ok else 60.0
    evidence = 100.0 * sum(event.verification_status == "verified" for event in events) / max(len(events), 1)
    named = [name for scene in scenes for name in scene.characters if name.upper() != "UNKNOWN"]
    unknown = [name for scene in scenes for name in scene.characters if name.upper() == "UNKNOWN"]
    character_score = 100.0 if not unknown else max(60.0, 100.0 - 4.0 * len(unknown) / max(len(named) + len(unknown), 1))
    budget = _review_narration_budget(target_minutes, style)
    word_count = _narration_word_count(segments)
    word_error = abs(word_count - budget.target_words) / max(budget.target_words, 1)
    duration_score = max(0.0, 100.0 - word_error * 250.0)

    role_order = {role: index for index, role in enumerate(_STORY_ROLE_ORDER)}
    observed_roles = [item.story_role for item in segments]
    role_values = [role_order.get(role, 999) for role in observed_roles]
    required_roles = set(_event_story_roles(events).values())
    arc_complete = required_roles.issubset(set(observed_roles))
    arc_monotonic = all(left <= right for left, right in zip(role_values, role_values[1:]))
    structural_story_score = 100.0 if arc_complete and arc_monotonic else 60.0 if arc_monotonic else 30.0
    story_score = (
        min(structural_story_score, narrative_assessment.coherence_score)
        if narrative_assessment is not None
        else structural_story_score
    )
    transition_issues = _narrative_transition_issues(segments)
    style_score = narrative_assessment.style_score if narrative_assessment is not None else 100.0
    if transition_issues:
        style_score = min(style_score, 84.0)
    source_coverage = _review_source_coverage_score(scenes, decisions, target_minutes)

    overall = (
        direct * 0.38
        + chronology * 0.16
        + evidence * 0.12
        + character_score * 0.07
        + duration_score * 0.08
        + story_score * 0.08
        + source_coverage * 0.06
        + style_score * 0.05
    )
    issues: list[QualityIssue] = []
    decision_by_segment = {decision.segment_id: decision for decision in decisions}
    for segment in segments:
        decision = decision_by_segment.get(segment.segment_id)
        if decision is None or not decision.source_clips:
            issues.append(
                QualityIssue(
                    severity="error",
                    code="NO_CLIP",
                    message="Câu chưa có cảnh minh họa.",
                    segment_id=segment.segment_id,
                )
            )
        elif decision.source_clips[0].match_score < 0.75:
            issues.append(
                QualityIssue(
                    severity="error",
                    code="LOW_VISUAL_MATCH",
                    message=f"Điểm khớp cảnh {decision.source_clips[0].match_score:.2f} dưới 0.75.",
                    segment_id=decision.segment_id,
                )
            )
        elif atomic_issues := _visual_atomicity_issues([segment], scenes):
            issues.append(
                QualityIssue(
                    severity="error",
                    code="UNSUPPORTED_NARRATION",
                    message=(
                        "Câu kể chứa chi tiết không được chính cảnh/subtitle nguồn xác nhận: "
                        + "; ".join(atomic_issues[:3])
                    )[:500],
                    segment_id=segment.segment_id,
                )
            )
    if chronology < 100:
        issues.append(
            QualityIssue(
                severity="error",
                code="CHRONOLOGY",
                message="EDL có event hoặc timestamp cảnh đi ngược thứ tự phim gốc.",
            )
        )
    if evidence < 95:
        issues.append(QualityIssue(severity="warning", code="UNVERIFIED_EVENT", message="Timeline còn event thiếu bằng chứng trực tiếp."))
    if not budget.min_words <= word_count <= budget.max_words:
        issues.append(
            QualityIssue(
                severity="error",
                code="NARRATION_DURATION_MISMATCH",
                message=(
                    f"Kịch bản có {word_count} từ; review {budget.target_minutes} phút cần "
                    f"{budget.min_words}-{budget.max_words} từ."
                ),
            )
        )
    if not arc_complete or not arc_monotonic:
        issues.append(
            QualityIssue(
                severity="error",
                code="STORY_ARC_INCOMPLETE",
                message="Lời review chưa đi đủ và đúng thứ tự hook, bối cảnh, xung đột, cao trào, kết.",
            )
        )
    if narrative_assessment is not None and not narrative_assessment.logic_passed:
        details = list(narrative_assessment.contradictions) + list(narrative_assessment.early_spoilers)
        detail = "; ".join(details[:3]) or narrative_assessment.feedback or "Mạch kể chưa đạt yêu cầu."
        issues.append(
            QualityIssue(
                severity="error",
                code="NARRATIVE_LOGIC",
                message=f"Logic kể chuyện chỉ đạt {narrative_assessment.coherence_score:.1f}/100: {detail[:500]}",
            )
        )
    if narrative_assessment is not None and not narrative_assessment.style_passed:
        issues.append(
            QualityIssue(
                severity="error",
                code="STYLE_MISMATCH",
                message=(
                    f"Phong cách {style} chỉ đạt {narrative_assessment.style_score:.1f}/100: "
                    f"{(narrative_assessment.feedback or 'Lời kể chưa thể hiện rõ phong cách đã chọn.')[:500]}"
                ),
            )
        )
    if transition_issues:
        issues.append(
            QualityIssue(
                severity="error",
                code="REPETITIVE_TRANSITIONS",
                message=(
                    "Lời review đang giống danh sách cảnh vì lặp từ nối: "
                    + " ".join(transition_issues[:3])
                )[:500],
            )
        )
    if source_coverage < 85.0:
        issues.append(
            QualityIssue(
                severity="error",
                code="SOURCE_COVERAGE",
                message=f"Kịch bản mới phủ {source_coverage:.1f}% timeline đã kiểm chứng.",
            )
        )
    return ReviewQualityReport(
        overall_score=round(overall, 2),
        direct_visual_match_percent=round(direct, 2),
        chronology_score=round(chronology, 2),
        evidence_score=round(evidence, 2),
        character_consistency_score=round(character_score, 2),
        duration_adherence_score=round(duration_score, 2),
        story_coherence_score=round(story_score, 2),
        source_coverage_score=round(source_coverage, 2),
        style_adherence_score=round(style_score, 2),
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
                    "required_visuals, SOURCE DIALOGUE/SUBTITLE và keyframe; chấm 0..1 xem cảnh có trực tiếp cho "
                    "thấy đúng nhân vật, hành động, "
                    "đồ vật và bối cảnh hay không. Cấm dùng kiến thức phim ngoài ảnh/evidence. "
                    "Trả JSON {segments:[{segment_id,direct_match,score,reason,best_keyframe_time}]}; "
                    "direct_match=true chỉ khi MỘT keyframe tốt nhất trực tiếp minh họa hành động trung tâm của câu; "
                    "không cộng dồn các hành động nằm ở nhiều keyframe cách xa nhau trong cùng scene. Nếu một keyframe "
                    "đã cho thấy đúng hành động trung tâm thì có thể true kể cả score thô hơi bảo thủ; "
                    "Nếu narration kể nhân vật nói/hỏi/giải thích/tiết lộ/đề nghị/cảnh báo/tranh cãi/quyết định, "
                    "SOURCE DIALOGUE/SUBTITLE của chính scene phải xác nhận trực tiếp ý nghĩa đó; nếu thiếu hoặc "
                    "mâu thuẫn thì direct_match=false kể cả keyframe có đúng nhân vật. "
                    "direct_match=false nếu sai chủ thể/hành động. best_keyframe_time phải là timestamp keyframe "
                    "đã được ghi nhãn nơi bằng chứng rõ nhất."
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
                        f"SOURCE DIALOGUE/SUBTITLE: {json.dumps(_scene_dialogue_evidence(scene), ensure_ascii=False)}\n"
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
        batch_succeeded = False
        for request_attempt in range(2):
            try:
                response = await _client().chat.completions.create(
                    model=settings.review_ai_model,
                    messages=[{"role": "user", "content": content}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    timeout=180,
                )
                batch_succeeded = True
                data = _loads_json(response.choices[0].message.content or "{}")
                values = data.get("segments", []) if isinstance(data, dict) else []
                if isinstance(values, list):
                    for item in values:
                        if not isinstance(item, dict):
                            continue
                        segment_id = str(item.get("segment_id") or "")
                        if segment_id in valid_ids:
                            score = _confidence(item.get("score"))
                            direct_match = item.get("direct_match")
                            if isinstance(direct_match, bool):
                                score = max(0.75, score) if direct_match else min(0.74, score)
                            ai_scores[segment_id] = (
                                score,
                                _safe_text(item.get("reason"), "Gemini xác nhận bằng keyframe và scene facts."),
                                _optional_number(item.get("best_keyframe_time")),
                            )
                if all(segment_id in ai_scores for segment_id in valid_ids):
                    break
            except Exception as exc:
                print(f"Pre-render multimodal rescoring fallback: {exc}")
                if request_attempt:
                    break
        if batch_succeeded:
            # A successful JSON response that omits one id is not a visual
            # pass. Retry once above, then block instead of retaining a stale
            # heuristic score that Gemini never verified.
            for segment_id in valid_ids:
                ai_scores.setdefault(
                    segment_id,
                    (0.0, "Gemini visual QA không trả kết quả cho segment sau hai lần thử.", None),
                )

    updated: list[EditDecision] = []
    for decision in decisions:
        value = ai_scores.get(decision.segment_id)
        if value is None or not decision.source_clips:
            updated.append(decision)
            continue
        ai_score, reason, best_time = value
        selected = decision.source_clips[0]
        score = _resolved_multimodal_score(selected.match_score, ai_score)
        scene = scene_by_id.get(selected.scene_id)
        start_seconds = selected.start_seconds
        end_seconds = selected.end_seconds
        if scene is not None and best_time is None:
            best_time = _verified_keyframe_time_from_text(reason, scene)
        if scene is not None and best_time is not None and scene.start_time <= best_time <= scene.end_time:
            start_seconds, end_seconds = _verified_clip_window(scene, best_time)
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


def _verified_keyframe_time_from_text(reason: str, scene: AnalyzedScene) -> float | None:
    """Recover a verified timestamp embedded in a gateway's reason text."""

    mentioned = [
        float(value.replace(",", "."))
        for value in re.findall(r"\d+(?:[.,]\d+)?", reason or "")
    ]
    if not mentioned or not scene.keyframes:
        return None
    matches = [
        keyframe.time
        for keyframe in scene.keyframes
        if any(abs(keyframe.time - value) <= 0.06 for value in mentioned)
    ]
    return matches[0] if matches else None


def _verified_clip_window(scene: AnalyzedScene, best_time: float) -> tuple[float, float]:
    scene_span = max(0.8, scene.end_time - scene.start_time)
    window = min(2.5, max(0.7, scene_span * 0.30))
    start = max(scene.start_time, min(best_time - window / 2.0, scene.end_time - window))
    return start, min(scene.end_time, start + window)


def recenter_edl_on_verified_keyframes(
    decisions: list[EditDecision],
    scenes: list[AnalyzedScene],
) -> list[EditDecision]:
    """Repair jobs saved before unit-suffixed keyframe times were accepted.

    The match reason persisted by older jobs still contains the exact verified
    keyframe.  Reusing it makes manual re-render deterministic and avoids an
    unnecessary second multimodal request for every sentence.
    """

    scene_by_id = {scene.scene_id: scene for scene in scenes}
    repaired: list[EditDecision] = []
    for decision in decisions:
        if not decision.source_clips:
            repaired.append(decision)
            continue
        selected = decision.source_clips[0]
        scene = scene_by_id.get(selected.scene_id)
        best_time = (
            _verified_keyframe_time_from_text(selected.match_reason, scene)
            if scene is not None
            else None
        )
        if scene is None or best_time is None:
            repaired.append(decision)
            continue
        start_seconds, end_seconds = _verified_clip_window(scene, best_time)
        centered = selected.model_copy(
            update={
                "start_seconds": round(start_seconds, 3),
                "end_seconds": round(end_seconds, 3),
            }
        )
        alternatives = [
            centered if item.candidate_id == centered.candidate_id else item
            for item in decision.alternatives
        ]
        repaired.append(
            decision.model_copy(
                update={"source_clips": [centered], "alternatives": alternatives}
            )
        )
    return repaired


async def _rescore_and_stabilize_timeline(
    segments: list[NarrationSegment],
    events: list[StoryEvent],
    scenes: list[AnalyzedScene],
    decisions: list[EditDecision],
    *,
    batch_size: int | None = None,
) -> tuple[list[NarrationSegment], list[EditDecision], bool]:
    """Apply visual QA and remove intra-event keyframe rewinds.

    Scene detection intentionally groups nearby shots.  Different actions in
    one broad scene therefore receive the same scene start until Gemini picks
    their exact keyframes.  If those keyframes reveal that the generated
    sentences are reversed, reorder the sentences (not the pictures) and
    rebuild the EDL so narration and visuals stay attached.
    """

    _ = events
    rescored = await _multimodal_rescore_selected_clips(
        segments,
        scenes,
        decisions,
        batch_size=batch_size,
    )
    reordered, changed = _reorder_intra_event_segments_by_verified_time(segments, rescored)
    if not changed:
        return segments, rescored, False
    # Every clip remains attached to the sentence/keyframe Gemini already
    # verified. Reordering those pairs needs no second 102-segment AI pass.
    return reordered, _reorder_decisions_to_segment_order(reordered, rescored), True


async def _rescore_changed_segments(
    segments: list[NarrationSegment],
    events: list[StoryEvent],
    scenes: list[AnalyzedScene],
    previous_decisions: list[EditDecision],
    changed_ids: set[str],
    *,
    batch_size: int | None = None,
) -> tuple[list[NarrationSegment], list[EditDecision], bool]:
    """Re-run expensive visual QA only for locally rewritten sentences."""

    rematched = _match_segments_to_scenes(segments, events, scenes)
    rematched_by_id = {item.segment_id: item for item in rematched}
    previous_by_id = {item.segment_id: item for item in previous_decisions}
    combined: list[EditDecision] = []
    for segment in segments:
        fresh = rematched_by_id.get(segment.segment_id)
        previous = previous_by_id.get(segment.segment_id)
        if fresh is None:
            continue
        if segment.segment_id in changed_ids or previous is None:
            combined.append(fresh)
        else:
            combined.append(
                previous.model_copy(
                    update={
                        "narration": segment.narration,
                        "voice_start": fresh.voice_start,
                        "voice_end": fresh.voice_end,
                    }
                )
            )

    changed_decisions = [item for item in combined if item.segment_id in changed_ids]
    if changed_decisions:
        changed_segments = [item for item in segments if item.segment_id in changed_ids]
        rescored_changed = await _multimodal_rescore_selected_clips(
            changed_segments,
            scenes,
            changed_decisions,
            batch_size=batch_size,
        )
        rescored_by_id = {item.segment_id: item for item in rescored_changed}
        combined = [rescored_by_id.get(item.segment_id, item) for item in combined]

    reordered, changed = _reorder_intra_event_segments_by_verified_time(segments, combined)
    if not changed:
        return segments, combined, False
    return reordered, _reorder_decisions_to_segment_order(reordered, combined), True


async def _repair_low_visual_until_stable(
    segments: list[NarrationSegment],
    events: list[StoryEvent],
    scenes: list[AnalyzedScene],
    decisions: list[EditDecision],
    style: str,
    target_minutes: int,
    assessment: NarrativeStyleAssessment,
    *,
    batch_size: int | None = None,
    character_names: CharacterNameRegistry | None = None,
) -> tuple[list[NarrationSegment], list[EditDecision], NarrativeStyleAssessment]:
    """Repair up to three small QA leftovers without rewriting good segments."""

    budget = _review_narration_budget(target_minutes, style)
    current_segments = segments
    current_decisions = decisions
    current_assessment = assessment
    for _ in range(3):
        proposed, changed = await _repair_low_visual_narration(
            current_segments,
            events,
            scenes,
            current_decisions,
            style,
            character_names=character_names,
        )
        if not changed:
            break
        if (
            not _narration_meets_budget(proposed, events, budget)
            or _visual_atomicity_issues(proposed, scenes)
            or _narrative_transition_issues(proposed)
        ):
            break
        local_assessment = await _judge_narrative_style(proposed, events, style)
        if (
            local_assessment.contradictions
            or local_assessment.early_spoilers
            or local_assessment.coherence_score + 5.0 < current_assessment.coherence_score
        ):
            break
        # A factual two-sentence repair cannot turn a previously accepted
        # 102-sentence style from 98 to 75.  The deterministic transition gate
        # above already blocks list-like prose, so retain the previously
        # verified style score while requiring fresh logic checks to pass.
        proposed_assessment = NarrativeStyleAssessment(
            coherence_score=min(current_assessment.coherence_score, local_assessment.coherence_score),
            style_score=current_assessment.style_score,
            contradictions=local_assessment.contradictions,
            early_spoilers=local_assessment.early_spoilers,
            feedback=current_assessment.feedback or local_assessment.feedback,
        )
        old_by_id = {item.segment_id: item for item in current_segments}
        changed_ids = {
            item.segment_id
            for item in proposed
            if old_by_id.get(item.segment_id) != item
        }
        if not changed_ids:
            break
        proposed, proposed_decisions, reordered = await _rescore_changed_segments(
            proposed,
            events,
            scenes,
            current_decisions,
            changed_ids,
            batch_size=batch_size,
        )
        if reordered:
            reordered_assessment = await _judge_narrative_style(proposed, events, style)
            if (
                reordered_assessment.contradictions
                or reordered_assessment.early_spoilers
                or reordered_assessment.coherence_score + 5.0 < proposed_assessment.coherence_score
            ):
                break
            proposed_assessment = NarrativeStyleAssessment(
                coherence_score=min(proposed_assessment.coherence_score, reordered_assessment.coherence_score),
                style_score=proposed_assessment.style_score,
                contradictions=reordered_assessment.contradictions,
                early_spoilers=reordered_assessment.early_spoilers,
                feedback=proposed_assessment.feedback or reordered_assessment.feedback,
            )
        current_segments = proposed
        current_decisions = proposed_decisions
        current_assessment = proposed_assessment
        if all(
            not decision.source_clips or decision.source_clips[0].match_score >= 0.75
            for decision in current_decisions
        ):
            break
    return current_segments, current_decisions, current_assessment


async def _repair_low_visual_narration(
    segments: list[NarrationSegment],
    events: list[StoryEvent],
    scenes: list[AnalyzedScene],
    decisions: list[EditDecision],
    style: str,
    *,
    character_names: CharacterNameRegistry | None = None,
) -> tuple[list[NarrationSegment], bool]:
    """Rewrite only sentences whose real keyframes cannot prove the claim."""

    segment_by_id = {item.segment_id: item for item in segments}
    event_by_id = {item.event_id: item for item in events}
    scene_by_id = {item.scene_id: item for item in scenes}
    low_decisions = [
        (index, decision)
        for index, decision in enumerate(decisions)
        if decision.source_clips and decision.source_clips[0].match_score < 0.75
    ]
    if not low_decisions:
        return segments, False

    proposed = list(segments)
    segment_index = {item.segment_id: index for index, item in enumerate(segments)}
    changed = False
    for offset in range(0, len(low_decisions), 3):
        batch = low_decisions[offset : offset + 3]
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "Sua cuc bo cac cau review bi keyframe QA tu choi. Moi segment chi chon MOT scene_id va MOT "
                    "action that su nhin thay ro trong it nhat mot keyframe. Tra JSON {segments:[{segment_id,scene_id,"
                    "narration,required_visuals:{characters,actions,objects,locations},confidence,purpose}]}. "
                    "required_visuals.actions co dung 1 action copy nguyen van tu SCENE FACTS cua scene da chon; "
                    "cac fact khac cung phai copy nguyen van. Narration la mot cau, chi ke action trung tam do, "
                    "giu dung event/phong cach. TARGET_WORDS chi la moc tham khao: neu cau cu ke thua chi tiet khong "
                    "co trong hinh thi uu tien cau moi gon va dung keyframe (khoang 50%-110% so tu cu), khong duoc "
                    "giu chi tiet sai chi de dem du tu. Khong suy dien hanh dong nam ngoai keyframe; cam dung "
                    "'va/nhung/roi' de noi them chu the hoac hanh dong thu hai. "
                    "Bat buoc DOI narration bi tu choi. Co the giu scene/action cu neu keyframe that su cho thay action "
                    "do va loi cu chi sai vi ke them chi tiet; khi ay cau moi chi duoc noi action nhin thay. Cam tra lai "
                    "nguyen van narration cu hoac chi doi metadata. Neu narration moi ke noi dung noi/hoi/giai thich/"
                    "tiet lo/de nghi/canh bao/tranh cai/quyet dinh thi dialogue hoac evidence.dialogue/subtitle cua "
                    "CHINH scene do phai truc tiep xac nhan; neu khong co sub/thoai phu hop thi chi mo ta hanh dong "
                    "nhin thay, cam suy dien noi dung loi noi."
                ),
            }
        ]
        if character_names:
            content[0]["text"] += (
                " Moi ten nhan vat phai chep dung CHARACTER_NAME_CONTRACT; khong dich ten va khong dung alias. "
                f"CHARACTER_NAME_CONTRACT={json.dumps(character_name_contract(character_names), ensure_ascii=False)}"
            )
        valid_ids: set[str] = set()
        decision_context: dict[str, tuple[int, float, float, set[str]]] = {}
        for decision_index, decision in batch:
            segment = segment_by_id.get(decision.segment_id)
            event = event_by_id.get(decision.event_id)
            if segment is None or event is None:
                continue
            candidate_by_scene: dict[str, SceneCandidate] = {}
            for candidate in decision.source_clips + decision.alternatives:
                if candidate.scene_id in scene_by_id:
                    candidate_by_scene.setdefault(candidate.scene_id, candidate)
            # The first EDL keeps only three candidates.  When all three are a
            # poor fit, the repair pass must still be able to choose another
            # verified scene from the same event instead of rewriting prose to
            # match an unrelated visual.
            for scene_id in event.scene_ids:
                scene = scene_by_id.get(scene_id)
                if scene is None or scene_id in candidate_by_scene:
                    continue
                score, reason = _scene_match_score(segment, event, scene)
                clip_duration = min(7.0, max(2.0, segment.estimated_voice_duration))
                start = min(
                    max(scene.start_time, scene.start_time + 0.15),
                    max(scene.start_time, scene.end_time - clip_duration),
                )
                end = min(scene.end_time, start + clip_duration)
                candidate_by_scene[scene_id] = SceneCandidate(
                    candidate_id=f"{segment.segment_id}:{scene_id}:repair",
                    scene_id=scene_id,
                    start_seconds=round(start, 3),
                    end_seconds=round(max(start + 0.8, end), 3),
                    thumbnail_path=scene.thumbnail_path,
                    match_score=round(score, 4),
                    match_reason=reason,
                )
            previous_start = next(
                (
                    earlier.source_clips[0].start_seconds
                    for earlier in reversed(decisions[:decision_index])
                    if earlier.source_clips
                ),
                float("-inf"),
            )
            next_start = next(
                (
                    later.source_clips[0].start_seconds
                    for later in decisions[decision_index + 1 :]
                    if later.source_clips
                ),
                float("inf"),
            )
            candidates = [
                item
                for item in candidate_by_scene.values()
                if item.start_seconds + 0.25 >= previous_start
                and item.start_seconds <= next_start + 0.25
            ]
            if not candidates:
                continue
            needs_dialogue = _narration_requires_dialogue_evidence(segment.narration)
            candidates.sort(
                key=lambda item: (
                    bool(_scene_dialogue_evidence(scene_by_id[item.scene_id]))
                    if needs_dialogue
                    else False,
                    item.match_score,
                ),
                reverse=True,
            )
            candidates = candidates[:6]
            valid_ids.add(segment.segment_id)
            decision_context[segment.segment_id] = (
                decision_index,
                previous_start,
                next_start,
                {item.scene_id for item in candidates},
            )
            word_count = len(re.findall(r"\w+", segment.narration, flags=re.UNICODE))
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"SEGMENT {segment.segment_id}; EVENT {segment.event_id}; STYLE {style}; "
                        f"TARGET_WORDS {word_count}; ALLOWED_SOURCE_TIME {previous_start:.2f}..{next_start:.2f}\n"
                        f"OLD: {segment.narration}\n"
                        f"CANDIDATE SCENES: {json.dumps([scene_by_id[item.scene_id].model_dump(exclude={'keyframes', 'thumbnail_path'}) for item in candidates], ensure_ascii=False)}"
                    ),
                }
            )
            for candidate in candidates:
                scene = scene_by_id[candidate.scene_id]
                for keyframe in scene.keyframes[:3]:
                    keyframe_path = Path(keyframe.path)
                    if not keyframe_path.exists():
                        continue
                    content.append(
                        {
                            "type": "text",
                            "text": f"{segment.segment_id} scene={scene.scene_id} time={keyframe.time:.2f}s",
                        }
                    )
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
                model=settings.review_ai_model,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0.1,
                timeout=180,
            )
            data = _loads_json(response.choices[0].message.content or "{}")
            values = data.get("segments", []) if isinstance(data, dict) else []
        except Exception as exc:
            print(f"Low-visual narration repair fallback: {exc}")
            continue
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            segment_id = str(item.get("segment_id") or "")
            original = segment_by_id.get(segment_id)
            context = decision_context.get(segment_id)
            if original is None or context is None:
                continue
            _, previous_start, next_start, allowed_scene_ids = context
            scene_id = str(item.get("scene_id") or "")
            scene = scene_by_id.get(scene_id)
            event = event_by_id.get(original.event_id)
            if scene is None or event is None or scene_id not in allowed_scene_ids or scene_id not in event.scene_ids:
                continue
            if scene.start_time + 0.25 < previous_start or scene.start_time > next_start + 0.25:
                continue
            narration = unicodedata.normalize("NFC", _safe_text(item.get("narration"), ""))
            required = _dict(item.get("required_visuals"))
            candidate = original.model_copy(
                update={
                    "narration": narration,
                    "required_visuals": RequiredVisuals(
                        characters=_string_list(required.get("characters")),
                        actions=_string_list(required.get("actions")),
                        objects=_string_list(required.get("objects")),
                        locations=_string_list(required.get("locations")),
                    ),
                    "candidate_scene_ids": [scene_id],
                    "estimated_voice_duration": _estimated_voice_duration(narration),
                    "confidence": _confidence(item.get("confidence")),
                    "purpose": _safe_text(item.get("purpose"), original.purpose),
                }
            )
            candidate = _canonicalize_narration_names(
                [candidate],
                character_names,
            )[0]
            original_words = len(re.findall(r"\w+", original.narration, flags=re.UNICODE))
            candidate_words = len(re.findall(r"\w+", candidate.narration, flags=re.UNICODE))
            min_repair_words = max(6, math.floor(original_words * 0.45))
            if candidate_words < min_repair_words or candidate_words > original_words + 4:
                continue
            if candidate.narration.casefold().strip() == original.narration.casefold().strip():
                continue
            if _visual_atomicity_issues([candidate], scene_by_id):
                continue
            proposed[segment_index[segment_id]] = candidate
            changed = True
    return proposed, changed


def _resolved_multimodal_score(previous_score: float, ai_score: float) -> float:
    """Use the frame-aware score without recursively blending old results.

    ``previous_score`` may already contain an earlier AI blend after a manual
    save. Blending it again makes the score drift every time the same scene is
    selected. The local score remains useful only when the multimodal request
    fails; once an AI score exists it is the authoritative visual result.
    """

    _ = previous_score
    return round(_confidence(ai_score), 4)


def _replace_low_pre_render_matches(
    decisions: list[EditDecision],
    segments: list[NarrationSegment] | None = None,
    scenes: list[AnalyzedScene] | None = None,
) -> tuple[list[EditDecision], bool]:
    """Try the next strong candidate when visual QA rejects the initial pick."""

    segment_by_id = {item.segment_id: item for item in segments or []}
    scene_by_id = {item.scene_id: item for item in scenes or []}

    def supports_sentence(candidate: SceneCandidate, decision: EditDecision) -> bool:
        if not segment_by_id or not scene_by_id:
            return True
        segment = segment_by_id.get(decision.segment_id)
        scene = scene_by_id.get(candidate.scene_id)
        return bool(segment and scene and _scene_supports_segment_required_visuals(segment, scene))

    changed = False
    updated: list[EditDecision] = []
    for index, decision in enumerate(decisions):
        current = decision.source_clips[0] if decision.source_clips else None
        if current is None or current.match_score >= 0.75:
            updated.append(decision)
            continue
        previous_start = (
            updated[-1].source_clips[0].start_seconds
            if updated and updated[-1].source_clips
            else float("-inf")
        )
        next_start = next(
            (
                later.source_clips[0].start_seconds
                for later in decisions[index + 1 :]
                if later.source_clips
            ),
            float("inf"),
        )
        alternative = next(
            (
                item
                for item in sorted(decision.alternatives, key=lambda value: value.match_score, reverse=True)
                if item.candidate_id != current.candidate_id and item.match_score >= 0.75
                and item.start_seconds + 0.25 >= previous_start
                and item.start_seconds <= next_start + 0.25
                and supports_sentence(item, decision)
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
    if not decisions or actual_duration <= 0:
        return decisions

    # The synthesized review is one continuous audio file, so there are no
    # per-sentence timestamps to copy back into the EDL. Allocate that measured
    # duration by narration length instead of scaling the old estimates: those
    # estimates are capped and often make a newly shortened sentence keep the
    # same screen time as a much longer one.
    weights = [max(1.0, float(len(" ".join(item.narration.split())))) for item in decisions]
    total_weight = sum(weights)
    result: list[EditDecision] = []
    cursor = 0.0
    cumulative_weight = 0.0
    for index, (decision, weight) in enumerate(zip(decisions, weights)):
        cumulative_weight += weight
        end = (
            float(actual_duration)
            if index == len(decisions) - 1
            else round(float(actual_duration) * cumulative_weight / total_weight, 3)
        )
        result.append(decision.model_copy(update={"voice_start": cursor, "voice_end": end}))
        cursor = end
    return result


def _post_render_segment_contract(
    decision: EditDecision,
    segment: NarrationSegment,
) -> str:
    """Describe only claims the rendered frame is actually required to prove."""

    central_action = segment.required_visuals.actions[0] if segment.required_visuals.actions else ""
    return (
        f"{decision.segment_id}\n"
        f"NARRATION: {decision.narration}\n"
        f"CENTRAL_ACTION_HINT: {central_action or '[none]'}"
    )


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
                    "Chỉ chấm điều NARRATION thực sự kể; CENTRAL_ACTION_HINT chỉ làm rõ hành động trung tâm. "
                    "Không được đòi thêm nhân vật, đồ vật hay địa điểm từ metadata nếu narration/action không nhắc tới. "
                    "Bỏ qua chữ phụ đề review phủ trên ảnh và không suy diễn từ kiến thức phim. "
                    "Cả ba frame phải tiếp tục minh họa trực tiếp cùng hành động hoặc bối cảnh mà câu kể. "
                    "Không bắt buộc giữ nguyên góc máy hay cùng một shot: chuyển từ toàn cảnh sang cận cảnh, "
                    "đổi góc quay hoặc camera cut vẫn hợp lệ nếu mọi frame đều trực tiếp liên quan đến câu. "
                    "Chủ thể hoặc hành động nằm ở hậu cảnh vẫn là bằng chứng trực tiếp nếu nhìn thấy rõ; "
                    "không được tự suy đoán đó là màn hình TV hay ảnh phản chiếu khi frame không chứng minh điều đó. "
                    "Chỉ coi là lẫn cảnh khi có frame chuyển sang hành động/bối cảnh không liên quan hoặc thiếu "
                    "chủ thể bắt buộc mà narration nhắc tới. "
                    "Trả JSON {segments:[{segment_id,direct_match,score,reason}]}, score 0..1. "
                    "direct_match=true và score >=0.75 khi chuỗi frame minh họa trực tiếp đúng câu; "
                    "direct_match=false và score <0.75 khi sai chủ thể/hành động/vật thể, có cảnh không liên quan, "
                    "frame đen hoặc không minh họa trực tiếp."
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
                    "text": _post_render_segment_contract(decision, segment),
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
        batch_succeeded = False
        for request_attempt in range(2):
            try:
                response = await _client().chat.completions.create(
                    model=settings.review_ai_model,
                    messages=[{"role": "user", "content": content}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    timeout=180,
                )
                batch_succeeded = True
                data = _loads_json(response.choices[0].message.content or "{}")
                values = data.get("segments", []) if isinstance(data, dict) else []
                if isinstance(values, list):
                    for item in values:
                        if not isinstance(item, dict):
                            continue
                        segment_id = str(item.get("segment_id") or "")
                        if segment_id not in valid_ids:
                            continue
                        score = _confidence(item.get("score"))
                        parsed_direct_match = _optional_bool(item.get("direct_match"))
                        direct_match = (
                            parsed_direct_match
                            if parsed_direct_match is not None
                            else score >= 0.75
                        )
                        if direct_match:
                            score = max(0.75, score)
                        else:
                            score = min(0.74, score)
                        scores[segment_id] = score
                        direct_matches[segment_id] = direct_match
                        notes[segment_id] = _safe_text(item.get("reason"), "Gemini không nêu lý do.")
                if all(segment_id in scores for segment_id in valid_ids):
                    break
            except Exception as exc:
                print(f"Post-render visual QA fallback: {exc}")
                if request_attempt:
                    break
        if batch_succeeded:
            for segment_id in valid_ids:
                if segment_id not in scores:
                    decision = next(item for item in batch if item.segment_id == segment_id)
                    fallback_score = (
                        decision.source_clips[0].match_score if decision.source_clips else 0.0
                    )
                    scores[segment_id] = fallback_score
                    direct_matches[segment_id] = fallback_score >= 0.75
                    notes[segment_id] = (
                        "Dùng điểm pre-render vì Gemini QA bỏ sót segment sau hai lần thử."
                    )
        else:
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
    target_duration_score, target_duration_ok = review_duration_adherence(
        media_duration,
        package.target_minutes,
    )
    overall = (
        direct * 0.43
        + chronology * 0.16
        + package.quality_report.evidence_score * 0.09
        + package.quality_report.character_consistency_score * 0.06
        + target_duration_score * 0.10
        + package.quality_report.story_coherence_score * 0.07
        + package.quality_report.source_coverage_score * 0.05
        + package.quality_report.style_adherence_score * 0.04
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
    if not target_duration_ok:
        requested_seconds = package.target_minutes * 60.0
        issues.append(
            QualityIssue(
                severity="error",
                code="TARGET_DURATION_MISMATCH",
                message=(
                    f"Video dài {media_duration:.1f} giây, lệch quá 10% so với mục tiêu "
                    f"{requested_seconds:.0f} giây ({package.target_minutes} phút)."
                ),
            )
        )
    if package.quality_report.story_coherence_score < 85.0:
        issues.append(
            QualityIssue(
                severity="error",
                code="NARRATIVE_LOGIC",
                message="Kịch bản chưa đạt cổng logic kể chuyện trước render.",
            )
        )
    if package.quality_report.style_adherence_score < 85.0:
        issues.append(
            QualityIssue(
                severity="error",
                code="STYLE_MISMATCH",
                message="Kịch bản chưa đạt phong cách review đã chọn.",
            )
        )
    report = ReviewQualityReport(
        phase="post_render",
        overall_score=round(overall, 2),
        direct_visual_match_percent=round(direct, 2),
        chronology_score=round(chronology, 2),
        evidence_score=package.quality_report.evidence_score,
        character_consistency_score=package.quality_report.character_consistency_score,
        duration_adherence_score=target_duration_score,
        story_coherence_score=package.quality_report.story_coherence_score,
        source_coverage_score=package.quality_report.source_coverage_score,
        style_adherence_score=package.quality_report.style_adherence_score,
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
            model=settings.review_ai_model,
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
    segments: list[NarrationSegment] | None = None,
    scenes: list[AnalyzedScene] | None = None,
) -> tuple[list[EditDecision], bool]:
    failed_issues = {
        issue.segment_id: issue for issue in report.issues if issue.segment_id
    }
    failed_ids = set(failed_issues)
    segment_by_id = {item.segment_id: item for item in segments or []}
    scene_by_id = {item.scene_id: item for item in scenes or []}

    def supports_sentence(candidate: SceneCandidate, decision: EditDecision) -> bool:
        if not segment_by_id or not scene_by_id:
            return True
        segment = segment_by_id.get(decision.segment_id)
        scene = scene_by_id.get(candidate.scene_id)
        return bool(segment and scene and _scene_supports_segment_required_visuals(segment, scene))

    changed = False
    updated: list[EditDecision] = []
    for decision in decisions:
        if decision.segment_id not in failed_ids:
            updated.append(decision)
            continue
        if decision.source_clips and scene_by_id:
            selected = decision.source_clips[0]
            scene = scene_by_id.get(selected.scene_id)
            stabilized = (
                _stabilize_failed_candidate_window(
                    selected,
                    scene,
                    failed_issues[decision.segment_id].message,
                )
                if scene is not None
                else None
            )
            if stabilized is not None:
                alternatives = [
                    stabilized if item.candidate_id == stabilized.candidate_id else item
                    for item in decision.alternatives
                ]
                updated.append(
                    decision.model_copy(
                        update={
                            "selected_candidate_id": stabilized.candidate_id,
                            "source_clips": [stabilized],
                            "alternatives": alternatives,
                        }
                    )
                )
                changed = True
                continue
        if len(decision.alternatives) < 2:
            updated.append(decision)
            continue
        current_id = decision.source_clips[0].candidate_id if decision.source_clips else ""
        alternative = next(
            (
                item
                for item in decision.alternatives
                if item.candidate_id != current_id and item.match_score >= 0.75
                and supports_sentence(item, decision)
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


def _stabilize_failed_candidate_window(
    candidate: SceneCandidate,
    scene: AnalyzedScene,
    issue_message: str,
) -> SceneCandidate | None:
    """Keep a verified action while removing the transition that QA saw.

    Multimodal matching verifies a keyframe, but a two-and-a-half-second clip
    around it can still cross into the preceding or following shot.  On the
    single automatic retry, retain the high-confidence candidate and take a
    one-second window on the safe side identified by the three-frame QA.  This
    is deterministic and avoids replacing a correct action with a weaker
    alternative scene.
    """

    current_duration = candidate.end_seconds - candidate.start_seconds
    if candidate.match_score < 0.75 or current_duration <= 1.05:
        return None
    best_time = _verified_keyframe_time_from_text(candidate.match_reason, scene)
    if best_time is None:
        return None

    message = " ".join((issue_message or "").casefold().split())
    early_problem = any(
        marker in message
        for marker in (
            "frame đầu",
            "frame thứ nhất",
            "frame 1",
            "khung hình đầu",
            "khung hình thứ nhất",
            "first frame",
        )
    )
    late_problem = any(
        marker in message
        for marker in (
            "frame cuối",
            "frame thứ ba",
            "frame 3",
            "khung hình cuối",
            "khung hình thứ ba",
            "hai frame cuối",
            "thứ hai và thứ ba",
            "last frame",
            "third frame",
        )
    )
    retry_duration = min(1.0, current_duration)
    lower = max(scene.start_time, candidate.start_seconds)
    upper = min(scene.end_time, candidate.end_seconds)
    if late_problem and not early_problem:
        end = min(upper, best_time)
        start = max(lower, end - retry_duration)
    elif early_problem and not late_problem:
        start = max(lower, best_time)
        end = min(upper, start + retry_duration)
    else:
        start = max(lower, min(best_time - retry_duration / 2.0, upper - retry_duration))
        end = min(upper, start + retry_duration)
    if end - start < 0.55:
        return None
    return candidate.model_copy(
        update={
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "match_reason": (
                f"{candidate.match_reason}; stabilized after post-render QA"
            ),
        }
    )


def _qa_frame_batches(frame_indices: list[int], batch_size: int = 32) -> list[list[int]]:
    size = max(1, min(32, int(batch_size)))
    return [frame_indices[index : index + size] for index in range(0, len(frame_indices), size)]


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
    ffmpeg = find_ffmpeg()
    index_to_path: dict[int, Path] = {}
    extraction_errors: list[str] = []
    # A single select expression with ~200 chained eq() calls exhausts the
    # FFmpeg expression parser on Windows ("Cannot allocate memory"). Decode
    # small batches instead; each output sequence remains easy to map back to
    # its exact requested frame index.
    frame_batches = _qa_frame_batches(frame_indices)
    for batch_index, batch in enumerate(frame_batches, start=1):
        if not ffmpeg:
            extraction_errors.append("FFmpeg is unavailable")
            break
        select_expression = "+".join(f"eq(n\\,{index})" for index in batch)
        output_pattern = output_dir / f"selected_{batch_index:03d}_%04d.jpg"
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
        selected_files = sorted(output_dir.glob(f"selected_{batch_index:03d}_*.jpg"))
        if completed.returncode != 0 or len(selected_files) != len(batch):
            detail = completed.stdout.decode(errors="replace")[-600:] if completed.stdout else ""
            extraction_errors.append(
                f"batch {batch_index}: FFmpeg code {completed.returncode}, "
                f"expected {len(batch)} frames, got {len(selected_files)}; {detail}"
            )
            continue
        index_to_path.update(zip(batch, selected_files))
    if extraction_errors:
        raise RuntimeError("Không trích được frame QA sau render. " + " | ".join(extraction_errors[:3]))
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
    # Multimodal gateways do not always honour the requested JSON scalar
    # type.  In practice ``best_keyframe_time`` is often returned as
    # ``"122.85s"`` or ``"122,85 giay"``.  Treating that as missing leaves
    # the EDL at the beginning of a broad consolidated scene even though the
    # model verified a much later keyframe.
    if isinstance(value, str):
        match = re.search(r"[-+]?\d+(?:[.,]\d+)?", value.strip())
        if not match:
            return None
        value = match.group(0).replace(",", ".")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_bool(value: object) -> bool | None:
    """Parse JSON booleans without treating the string ``"false"`` as true."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    return None


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
