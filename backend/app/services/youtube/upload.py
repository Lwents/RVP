from googleapiclient.http import MediaFileUpload
from app.services.youtube.auth import get_youtube_client

def upload_video_to_youtube(video_path: str, title: str, description: str, tags: list[str], privacy_status: str = "private") -> dict:
    """Tải video lên YouTube và trả về thông tin video."""
    youtube = get_youtube_client()
    
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "24"  # Entertainment
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False, 
        }
    }
    
    # Media file upload
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    
    request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=media
    )
    
    response = None
    print(f"Uploading video {title}...")
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploaded {int(status.progress() * 100)}%")
            
    print(f"Upload Complete! Video ID: {response['id']}")
    return response

def get_channel_videos_stats():
    """Lấy số liệu thống kê (views, likes) của các video đã upload trên kênh."""
    youtube = get_youtube_client()
    
    # 1. Lấy Uploads playlist ID của channel
    channel_response = youtube.channels().list(
        part="contentDetails",
        mine=True
    ).execute()
    
    if not channel_response.get("items"):
        return []
        
    uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    
    # 2. Lấy danh sách video ID trong playlist uploads
    playlist_response = youtube.playlistItems().list(
        part="contentDetails",
        playlistId=uploads_playlist_id,
        maxResults=50
    ).execute()
    
    video_ids = [item["contentDetails"]["videoId"] for item in playlist_response.get("items", [])]
    if not video_ids:
        return []
        
    # 3. Lấy stats cho các video ID này
    videos_response = youtube.videos().list(
        part="snippet,statistics",
        id=",".join(video_ids)
    ).execute()
    
    stats = []
    for video in videos_response.get("items", []):
        stats.append({
            "id": video["id"],
            "title": video["snippet"]["title"],
            "thumbnail": video["snippet"]["thumbnails"].get("medium", {}).get("url", ""),
            "published_at": video["snippet"]["publishedAt"],
            "views": video["statistics"].get("viewCount", "0"),
            "likes": video["statistics"].get("likeCount", "0"),
            "comments": video["statistics"].get("commentCount", "0"),
        })
        
    return stats
