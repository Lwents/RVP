import json
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from app.core import settings

class VideoDetails(BaseModel):
    title: str = Field(description="Tiêu đề video giật tít, thu hút người xem (tối đa 100 ký tự).")
    description: str = Field(description="Mô tả video chi tiết, hấp dẫn, bao gồm tóm tắt nội dung.")
    tags: list[str] = Field(description="Danh sách các tags liên quan đến video, ví dụ: ['hoathinh', 'review'].")

async def generate_video_details(transcript: str, custom_prompt: str | None = None) -> VideoDetails:
    """
    Sử dụng AI qua 9router để tạo chi tiết video (tiêu đề, mô tả, tags).
    """
    client = AsyncOpenAI(
        api_key=settings.ninerouter_api_key,
        base_url=settings.ninerouter_api_url,
    )

    system_prompt = (
        "Bạn là một chuyên gia SEO YouTube chuyên tạo content thu hút người xem. "
        "Dựa vào nội dung phụ đề (transcript) sau đây, hãy tạo ra 1 Tiêu đề (dưới 100 ký tự), "
        "1 Mô tả tóm tắt nội dung hấp dẫn, và 1 danh sách các Tags phù hợp. "
        "Trả về ĐỊNH DẠNG JSON với các key: 'title', 'description', 'tags'."
    )
    
    if custom_prompt:
        system_prompt += f"\nYêu cầu đặc biệt từ người dùng: {custom_prompt}"

    try:
        response = await client.chat.completions.create(
            model="ag/gemini-3.5-flash-low",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript của video:\n{transcript[:8000]}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from AI")
            
        data = json.loads(content)
        
        return VideoDetails(
            title=data.get("title", "Video Mới"),
            description=data.get("description", "Mô tả đang cập nhật..."),
            tags=data.get("tags", [])
        )
    except Exception as e:
        print(f"Error generating video details with AI: {e}")
        # Fallback values
        return VideoDetails(
            title="Tập 1 - Review Phim Mới",
            description="Tóm tắt: " + transcript[:200] + "...",
            tags=["review", "phim"]
        )
