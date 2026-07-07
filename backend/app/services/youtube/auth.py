import os
import json
from pathlib import Path
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from app.core import settings

SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube.readonly"]

def get_youtube_auth_url(redirect_uri: str = "http://localhost:5173/youtube/callback") -> str:
    """Trả về URL xác thực OAuth2 cho người dùng."""
    client_secrets_file = Path(settings.youtube_client_secrets_file)
    if not client_secrets_file.exists():
        raise FileNotFoundError(f"Chưa có file {settings.youtube_client_secrets_file}. Vui lòng tạo trên Google Cloud Console.")

    # Enable insecure transport for localhost testing
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

    flow = Flow.from_client_secrets_file(
        str(client_secrets_file),
        scopes=SCOPES,
        redirect_uri=redirect_uri
    )
    auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
    return auth_url

def handle_oauth2_callback(code: str, redirect_uri: str = "http://localhost:5173/youtube/callback") -> dict:
    """Xử lý code từ callback và lưu thông tin đăng nhập."""
    client_secrets_file = Path(settings.youtube_client_secrets_file)
    if not client_secrets_file.exists():
        raise FileNotFoundError("Chưa có file client_secrets.json")

    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

    flow = Flow.from_client_secrets_file(
        str(client_secrets_file),
        scopes=SCOPES,
        redirect_uri=redirect_uri
    )
    flow.fetch_token(code=code)
    credentials = flow.credentials

    # Lưu token vào file
    creds_data = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }
    
    creds_path = Path(settings.youtube_credentials_file)
    creds_path.parent.mkdir(parents=True, exist_ok=True)
    creds_path.write_text(json.dumps(creds_data, indent=2))

    return {"status": "success", "message": "Đã xác thực YouTube thành công"}

def get_youtube_client():
    """Lấy client YouTube API đã được xác thực."""
    creds_path = Path(settings.youtube_credentials_file)
    if not creds_path.exists():
        raise FileNotFoundError("Chưa xác thực YouTube. Vui lòng kết nối tài khoản.")

    creds_data = json.loads(creds_path.read_text())
    credentials = Credentials.from_authorized_user_info(creds_data, SCOPES)
    
    # Refresh token if needed
    if credentials.expired and credentials.refresh_token:
        from google.auth.transport.requests import Request
        credentials.refresh(Request())
        # Lưu lại token mới
        creds_data['token'] = credentials.token
        creds_path.write_text(json.dumps(creds_data, indent=2))

    return build("youtube", "v3", credentials=credentials)
