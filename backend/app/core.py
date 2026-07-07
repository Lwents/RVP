from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    cors_origins: list[str] = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]
    storage_dir: str = "storage"
    ai_engine: str = "passthrough"
    translation_engine: str = "passthrough"
    voice_engine: str = "disabled"
    target_language: str = "vi"
    ffmpeg_path: str | None = None
    video_encoder: str = "h264_nvenc"
    video_crf: int = 23
    video_preset: str = "slow"
    ytdlp_format: str = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo*+bestaudio/best"
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
    ninerouter_api_key: str = "sk-placeholder"
    youtube_client_secrets_file: str = "client_secret.json"
    youtube_credentials_file: str = "storage/youtube_credentials.json"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AUTO_TRANSLATE_", extra="ignore")


settings = Settings()
