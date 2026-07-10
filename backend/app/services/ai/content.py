import json
import re
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from app.core import settings

class VideoDetails(BaseModel):
    title: str = Field(description="Tiêu đề YouTube tiếng Việt, hấp dẫn nhưng không sai sự thật.")
    description: str = Field(description="Mô tả YouTube chuyên nghiệp, có hook, tóm tắt, CTA và hashtag.")
    tags: list[str] = Field(description="Danh sách tag SEO không có dấu #, ví dụ: ['reviewphim', 'tomtatphim'].")


class ReviewBeat(BaseModel):
    time_hint: str = Field(description="Moc thoi gian uoc luong, vi du 00:12:30-00:13:20.")
    purpose: str = Field(description="Vai tro cua canh trong video review.")
    narration: str = Field(description="Loi dan ngan gan voi canh nay.")


class MovieReviewPlan(BaseModel):
    title: str
    target_minutes: int
    hook: str
    summary: str
    narration_script: str
    beats: list[ReviewBeat]
    thumbnail_text: str
    tags: list[str]


async def generate_movie_review_plan(
    transcript: str,
    target_minutes: int = 8,
    style: str = "story",
    custom_prompt: str | None = None,
) -> MovieReviewPlan:
    clean_transcript = _plain_transcript(transcript)
    client = AsyncOpenAI(
        api_key=settings.ninerouter_api_key,
        base_url=settings.ninerouter_api_url,
    )

    style_map = {
        "story": "ke chuyen cuon hut, ro mach nhan vat va bien co",
        "fast": "nhanh, gon, nhieu hook, hop video 3-5 phut",
        "emotional": "cam xuc, nhan vao hy sinh, bi kich va cao trao",
        "funny": "duyen, nhe nhang, co chut hai nhung khong pha nat noi dung",
    }
    style_text = style_map.get(style, style_map["story"])
    prompt = (
        "Ban la bien tap vien kenh review phim tieng Viet. Hay bien transcript phim dai thanh mot ban review co the dung de dung video.\n"
        f"Muc tieu do dai: {target_minutes} phut. Phong cach: {style_text}.\n"
        "Yeu cau:\n"
        "- Khong bia dat ngoai noi dung transcript.\n"
        "- Viet loi dan tieng Viet tu nhien, giong nguoi review phim.\n"
        "- Chia thanh cac beat/canh de editor cat ghep minh hoa.\n"
        "- Moi beat can co time_hint, purpose va narration.\n"
        "- narration_script phai doc lien mach duoc, khong chi la dan y.\n"
        "- Tra ve dung JSON voi key: title, target_minutes, hook, summary, narration_script, beats, thumbnail_text, tags.\n"
    )
    if custom_prompt:
        prompt += f"\nYeu cau rieng: {custom_prompt}\n"

    try:
        response = await client.chat.completions.create(
            model="ag/gemini-3.5-flash-low",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Transcript phim:\n{clean_transcript[:18000]}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.75,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from AI")
        data = _loads_json_object(content)
        beats = data.get("beats", [])
        if not isinstance(beats, list):
            beats = []
        cleaned_tags = _clean_tags(data.get("tags", [])) or _default_tags()
        return MovieReviewPlan(
            title=_polish_title(str(data.get("title") or _fallback_title(clean_transcript))),
            target_minutes=max(1, min(30, int(data.get("target_minutes") or target_minutes))),
            hook=str(data.get("hook") or "Mot cau chuyen bat dau bang bien co khien moi thu dao lon."),
            summary=str(data.get("summary") or clean_transcript[:500]),
            narration_script=str(data.get("narration_script") or _fallback_description(clean_transcript)),
            beats=[
                ReviewBeat(
                    time_hint=str(item.get("time_hint") or "auto"),
                    purpose=str(item.get("purpose") or "Canh minh hoa noi dung chinh."),
                    narration=str(item.get("narration") or ""),
                )
                for item in beats[:18]
                if isinstance(item, dict)
            ],
            thumbnail_text=str(data.get("thumbnail_text") or "Cai ket khong ai ngo"),
            tags=cleaned_tags,
        )
    except Exception as exc:
        print(f"Error generating movie review plan with AI: {exc}")
        return MovieReviewPlan(
            title=_fallback_title(clean_transcript),
            target_minutes=target_minutes,
            hook="Mot cau chuyen bat dau bang bien co lon, day nhan vat vao lua chon kho khan.",
            summary=clean_transcript[:700] or "Chua co transcript du de tom tat.",
            narration_script=_fallback_description(clean_transcript),
            beats=[
                ReviewBeat(time_hint="00:00:00-00:01:00", purpose="Mo dau va dat van de", narration="Mo dau cau chuyen va gioi thieu xung dot chinh."),
                ReviewBeat(time_hint="auto", purpose="Cao trao", narration="Chon cac canh co bien co lon de day nhip review."),
                ReviewBeat(time_hint="auto", purpose="Ket", narration="Tom lai cai ket va dat cau hoi keo binh luan."),
            ],
            thumbnail_text="Cai ket khong ai ngo",
            tags=_default_tags(),
        )

async def generate_video_details(transcript: str, custom_prompt: str | None = None) -> VideoDetails:
    """
    Sử dụng AI qua 9router để tạo chi tiết video (tiêu đề, mô tả, tags).
    """
    clean_transcript = _plain_transcript(transcript)

    client = AsyncOpenAI(
        api_key=settings.ninerouter_api_key,
        base_url=settings.ninerouter_api_url,
    )

    system_prompt = (
        "Bạn là một YouTuber chuyên nghiệp kiêm chuyên gia SEO cho kênh review phim/hoạt hình tiếng Việt. "
        "Hãy viết metadata như người làm YouTube lâu năm: tự nhiên, cuốn, có cảm xúc, không giật tít sai sự thật.\n\n"
        "Yêu cầu title:\n"
        "- Tiếng Việt, 55-85 ký tự, không hashtag.\n"
        "- Có hook mạnh ở đầu, nêu đúng cảm xúc/xung đột chính của video.\n"
        "- Không dùng chữ in hoa toàn bộ, không spam dấu chấm than.\n\n"
        "Yêu cầu description:\n"
        "- Viết theo format chuyên nghiệp, dễ copy lên YouTube.\n"
        "- Dòng 1 là hook ngắn khiến người xem muốn xem hết.\n"
        "- Đoạn 2 tóm tắt nội dung chính trong 2-3 câu, không spoil quá đà nếu không cần.\n"
        "- Có 1 câu hỏi kéo bình luận.\n"
        "- Có CTA nhẹ: đăng ký/kênh/bình luận/chia sẻ.\n"
        "- Cuối mô tả có một dòng hashtag 5-8 hashtag liên quan.\n\n"
        "Yêu cầu tags:\n"
        "- 15-20 tag SEO không có dấu #, không có khoảng trắng.\n"
        "- Ưu tiên: reviewphim, tomtatphim, hoathinh, donghua, phimcotrang, phimnguoctam, tên nhân vật/sự kiện nếu nhận ra.\n\n"
        "Trả về đúng JSON với key: title, description, tags."
    )
    
    if custom_prompt:
        system_prompt += f"\nYêu cầu đặc biệt từ người dùng: {custom_prompt}"

    try:
        response = await client.chat.completions.create(
            model="ag/gemini-3.5-flash-low",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript của video:\n{clean_transcript[:8000]}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from AI")
            
        data = _loads_json_object(content)
        
        tags = data.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        cleaned_tags = _clean_tags(tags) or _default_tags()

        return VideoDetails(
            title=_polish_title(str(data.get("title") or "Review phim hoạt hình cảm động")),
            description=_polish_description(str(data.get("description") or ""), cleaned_tags),
            tags=cleaned_tags
        )
    except Exception as e:
        print(f"Error generating video details with AI: {e}")
        # Fallback values
        return VideoDetails(
            title=_fallback_title(clean_transcript),
            description=_fallback_description(clean_transcript),
            tags=_default_tags()
        )


def _loads_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _plain_transcript(transcript: str) -> str:
    lines: list[str] = []
    for raw in transcript.splitlines():
        line = raw.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _fallback_title(transcript: str) -> str:
    if not transcript:
        return "Câu chuyện cảm động khiến người xem nghẹn lòng"
    first_sentence = re.split(r"(?<=[.!?。！？])\s+", transcript)[0]
    first_sentence = first_sentence[:58].strip(" ,.;:-")
    return _polish_title(f"Cảnh phim khiến ai xem cũng nghẹn lòng: {first_sentence}")


def _fallback_description(transcript: str) -> str:
    summary = transcript[:260].strip()
    if not summary:
        summary = "Một câu chuyện nhiều cảm xúc được tóm tắt và lồng tiếng lại bằng tiếng Việt."
    return _polish_description(
        (
            "Một phân cảnh nhiều cảm xúc, càng xem càng thấy nghẹn.\n\n"
            f"Trong video này, mình tóm tắt lại nội dung chính: {summary}...\n\n"
            "Bạn thấy đoạn nào lấy cảm xúc nhất? Bình luận cảm nhận của bạn bên dưới nhé.\n"
            "Nếu thích kiểu review này, hãy đăng ký kênh để xem thêm những câu chuyện hay hơn."
        ),
        _default_tags(),
    )


def _clean_tags(tags: list[object]) -> list[str]:
    cleaned: list[str] = []
    for tag in tags:
        value = str(tag).strip().lower().lstrip("#")
        value = "".join(ch for ch in value if ch.isalnum() or ch in {"_", "-"})
        if not value or value in cleaned:
            continue
        cleaned.append(value)
        if len(cleaned) >= 18:
            break
    return cleaned


def _polish_title(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", title).strip(" \n\t\"'")
    cleaned = cleaned.replace("!!!", "!").replace("!!", "!")
    if not cleaned:
        cleaned = "Câu chuyện cảm động khiến người xem nghẹn lòng"
    if len(cleaned) > 85:
        cleaned = cleaned[:82].rstrip(" ,.;:-") + "..."
    return cleaned


def _polish_description(description: str, tags: list[str]) -> str:
    text = re.sub(r"\n{3,}", "\n\n", description.strip())
    if not text:
        text = (
            "Một phân cảnh nhiều cảm xúc, càng xem càng thấy nghẹn.\n\n"
            "Trong video này, mình tóm tắt lại câu chuyện theo cách dễ hiểu, cuốn và giữ đúng tinh thần nội dung gốc.\n\n"
            "Bạn thấy đoạn nào đáng nhớ nhất? Bình luận cảm nhận của bạn bên dưới nhé.\n"
            "Nếu thích kiểu review này, hãy đăng ký kênh để xem thêm những câu chuyện hay hơn."
        )

    hashtag_line = _hashtag_line(tags)
    if hashtag_line and "#" not in text.splitlines()[-1]:
        text = f"{text.rstrip()}\n\n{hashtag_line}"
    return text


def _hashtag_line(tags: list[str]) -> str:
    selected = tags[:8] if tags else _default_tags()[:8]
    return " ".join(f"#{tag}" for tag in selected)


def _default_tags() -> list[str]:
    return [
        "reviewphim",
        "tomtatphim",
        "hoathinh",
        "donghua",
        "phimcotrang",
        "phimnguoctam",
        "longtiengviet",
        "phimhay",
        "reviewhoathinh",
        "phimcamdong",
        "cotrangtrungquoc",
        "xuhuong",
        "viralvideo",
        "storytelling",
        "reviewyoutube",
    ]
