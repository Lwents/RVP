from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_AI_MODEL = "ag/gemini-3.8-flash-medium"
DEFAULT_REVIEW_AI_MODEL = "ag/gemini-pro-agent"


class Settings(BaseSettings):
    cors_origins: list[str] = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:8100",
        "http://localhost:8100",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ]
    storage_dir: str = str(BACKEND_DIR / "storage")
    ai_engine: str = "passthrough"
    translation_engine: str = "gemini"
    voice_engine: str = "disabled"
    target_language: str = "vi"
    ffmpeg_path: str | None = None
    video_encoder: str = "h264_nvenc"
    video_crf: int = 23
    video_preset: str = "slow"
    ytdlp_format: str = "bestvideo[height<=1080][ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best"
    ytdlp_download_timeout_seconds: int = 1800
    ytdlp_js_runtime: str | None = None
    ytdlp_cookies_file: str | None = None
    ytdlp_cookies_from_browser: str | None = None
    ytdlp_auto_browser_cookies: bool = True
    ytdlp_browser_cookie_sources: str = "edge,chrome"
    douyin_downloader_tool_enabled: bool = True
    douyin_downloader_tool_path: str | None = "../tools/douyin-downloader"
    douyin_external_downloader_enabled: bool = True
    douyin_external_downloader_services: str = "unduhtiktok,douyinwtf,tikwm"
    douyin_external_downloader_custom_urls: str | None = None
    douyin_external_downloader_timeout_seconds: int = 45
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5
    whisper_vad_filter: bool = True
    whisper_word_timestamps: bool = True
    whisper_local_files_only: bool = False
    prefer_youtube_subtitles: bool = True
    asr_timeout_seconds: int = 300
    tts_chunk_chars: int = 900
    tts_chunk_timeout_seconds: int = 120
    tts_chunk_retries: int = 3
    render_without_tts_on_error: bool = False
    subtitle_group_max_chars: int = 70
    subtitle_group_max_duration: float = 4.5
    subtitle_group_max_gap: float = 0.75
    
    ninerouter_api_url: str = "http://localhost:20128/v1"
    ninerouter_api_key: str | None = None
    # One source of truth for every 9router/Gemini feature in this project.
    ai_model: str = DEFAULT_AI_MODEL
    # Review phim needs stronger long-context visual/narrative reasoning than
    # short subtitle translation, so it has an independent model setting.
    review_ai_model: str = DEFAULT_REVIEW_AI_MODEL
    review_scene_batch_size: int = 4
    review_keyframes_per_scene: int = 3
    review_max_scenes: int = 240
    review_quality_threshold: int = 90

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_prefix="AUTO_TRANSLATE_",
        extra="ignore",
    )


settings = Settings()
