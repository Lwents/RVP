import React, { useEffect, useState, useRef } from "react";
import { Loader2, Upload, ExternalLink } from "lucide-react";
import { getYoutubeAuthUrl, getYoutubeStats, sendYoutubeCallbackCode, uploadYoutubeClientSecret } from "../lib/api";

// Global flag to prevent double-exchange race conditions in React 18 Strict Mode
let oauthExchangeInitiated = false;

export const YoutubeStats = React.memo(function YoutubeStats() {
  const [channel, setChannel] = useState<any | null>(null);
  const [videos, setVideos] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadingSecret, setUploadingSecret] = useState(false);

  const redirectUri = window.location.origin + "/youtube/callback";

  const parseErrorMessage = (errMessage: string): string => {
    try {
      const parsed = JSON.parse(errMessage);
      if (parsed && typeof parsed === "object" && parsed.detail) {
        return parsed.detail;
      }
    } catch (e) {
      // Message is not a JSON string, return as is
    }
    return errMessage;
  };

  useEffect(() => {
    // Check if we are returning from OAuth callback
    const urlParams = new URLSearchParams(window.location.search);
    const code = urlParams.get("code");
    
    if (code) {
      if (!oauthExchangeInitiated) {
        oauthExchangeInitiated = true;
        setLoading(true);
        
        // Clean URL immediately
        window.history.replaceState({}, document.title, window.location.pathname);
        
        sendYoutubeCallbackCode(code, redirectUri)
          .then(() => {
            oauthExchangeInitiated = false;
            fetchStats();
          })
          .catch(err => {
            oauthExchangeInitiated = false;
            setError(parseErrorMessage(err.message || "Failed to authenticate with YouTube"));
            setLoading(false);
          });
      }
    } else {
      if (!oauthExchangeInitiated) {
        fetchStats();
      }
    }
  }, []);

  const fetchStats = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getYoutubeStats();
      setChannel(data.channel);
      setVideos(data.videos || []);
    } catch (err: any) {
      const rawMsg = err.message || "Lỗi tải dữ liệu. Vui lòng xác thực kênh Youtube.";
      setError(parseErrorMessage(rawMsg));
    } finally {
      setLoading(false);
    }
  };

  const handleAuth = async () => {
    try {
      const res = await getYoutubeAuthUrl(redirectUri);
      window.location.href = res.url;
    } catch (err: any) {
      setError(parseErrorMessage(err.message || "Failed to get auth URL"));
    }
  };

  const handleSecretUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadingSecret(true);
    setError(null);
    try {
      await uploadYoutubeClientSecret(file);
      setError(null);
      fetchStats();
    } catch (err: any) {
      setError(err.message || "Không thể tải lên tệp client_secret.json");
    } finally {
      setUploadingSecret(false);
      e.target.value = "";
    }
  };

  const formatNumber = (numStr: string | number): string => {
    if (numStr === undefined || numStr === null) return "0";
    const num = typeof numStr === "string" ? parseInt(numStr, 10) : numStr;
    if (isNaN(num)) return "0";
    return num.toLocaleString();
  };

  const formatDuration = (durationStr: string): string => {
    if (!durationStr) return "";
    const matches = durationStr.match(/PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?/);
    if (!matches) return durationStr;
    const hours = parseInt(matches[1] || "0");
    const minutes = parseInt(matches[2] || "0");
    const seconds = parseInt(matches[3] || "0");
    
    let result = "";
    if (hours > 0) {
      result += hours + ":";
      result += (minutes < 10 ? "0" : "") + minutes + ":";
    } else {
      result += minutes + ":";
    }
    result += (seconds < 10 ? "0" : "") + seconds;
    return result;
  };

  const getPrivacyBadge = (privacy: string) => {
    let label = "Công khai";
    let color = "#34c759";
    let bg = "rgba(52, 199, 89, 0.1)";

    switch (privacy?.toLowerCase()) {
      case "private":
        label = "Riêng tư";
        color = "#ff3b30";
        bg = "rgba(255, 59, 48, 0.1)";
        break;
      case "unlisted":
        label = "Không công khai";
        color = "#ff9500";
        bg = "rgba(255, 149, 0, 0.1)";
        break;
    }

    return (
      <span style={{ 
        color, 
        backgroundColor: bg, 
        padding: '2px 8px', 
        borderRadius: '6px', 
        fontSize: '0.8rem', 
        fontWeight: 600,
        display: 'inline-block'
      }}>
        {label}
      </span>
    );
  };

  return (
    <div className="ios-section ios-container" style={{ maxWidth: '900px' }}>
      <div className="ios-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '32px' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 600 }}>Quản lý Kênh YouTube</h2>
          <button className="ios-button" onClick={handleAuth}>
            Kết nối / Cập nhật token
          </button>
        </div>

        {error && (
          <div style={{ backgroundColor: 'rgba(255, 59, 48, 0.1)', color: '#ff3b30', padding: '20px', borderRadius: '16px', marginBottom: '24px', fontWeight: 500, border: '1px solid rgba(255, 59, 48, 0.2)' }}>
            <div style={{ lineHeight: 1.5 }}>{error}</div>
            {error.includes("client_secret.json") && (
              <div style={{ marginTop: '16px' }}>
                <button 
                  className="ios-button" 
                  type="button" 
                  onClick={() => document.getElementById('secret-file-input')?.click()}
                  disabled={uploadingSecret}
                  style={{ display: "flex", alignItems: "center", gap: "8px" }}
                >
                  {uploadingSecret ? <Loader2 className="spin" size={17} /> : <Upload size={17} />}
                  <span>Chọn và tải lên file client_secret.json</span>
                </button>
                <input 
                  id="secret-file-input" 
                  type="file" 
                  accept=".json" 
                  onChange={handleSecretUpload} 
                  style={{ display: "none" }} 
                />
              </div>
            )}
          </div>
        )}

        {loading ? (
          <div style={{ textAlign: 'center', padding: '40px', color: '#86868b', fontWeight: 500 }}>Đang tải dữ liệu...</div>
        ) : (
          <div>
            {channel && (
              <div className="ios-glass" style={{ padding: '24px', borderRadius: '24px', marginBottom: '32px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                  <img src={channel.avatar} alt={channel.title} style={{ width: '64px', height: '64px', borderRadius: '50%', border: '2px solid #007aff', objectFit: 'cover' }} />
                  <div>
                    <h3 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 600, color: '#1d1d1f' }}>{channel.title}</h3>
                    <span style={{ fontSize: '0.9rem', color: '#86868b', fontWeight: 500 }}>Kênh YouTube đã liên kết</span>
                  </div>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                  <div style={{ backgroundColor: 'rgba(255,255,255,0.5)', padding: '16px', borderRadius: '16px', textAlign: 'center', border: '1px solid rgba(0,0,0,0.03)' }}>
                    <div style={{ fontSize: '0.85rem', color: '#86868b', fontWeight: 500, marginBottom: '6px' }}>Người đăng ký</div>
                    <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#007aff' }}>{formatNumber(channel.subscribers)}</div>
                  </div>
                  <div style={{ backgroundColor: 'rgba(255,255,255,0.5)', padding: '16px', borderRadius: '16px', textAlign: 'center', border: '1px solid rgba(0,0,0,0.03)' }}>
                    <div style={{ fontSize: '0.85rem', color: '#86868b', fontWeight: 500, marginBottom: '6px' }}>Tổng lượt xem</div>
                    <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#1d1d1f' }}>{formatNumber(channel.views)}</div>
                  </div>
                  <div style={{ backgroundColor: 'rgba(255,255,255,0.5)', padding: '16px', borderRadius: '16px', textAlign: 'center', border: '1px solid rgba(0,0,0,0.03)' }}>
                    <div style={{ fontSize: '0.85rem', color: '#86868b', fontWeight: 500, marginBottom: '6px' }}>Tổng số video</div>
                    <div style={{ fontSize: '1.2rem', fontWeight: 700, color: '#1d1d1f' }}>{formatNumber(channel.videos_count)}</div>
                  </div>
                </div>
              </div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {videos.length === 0 && !error ? (
                <div style={{ textAlign: 'center', padding: '40px', color: '#86868b', fontWeight: 500 }}>Không có video nào được tìm thấy.</div>
              ) : (
                videos.map(video => (
                  <div key={video.id} className="yt-stat-item ios-glass" style={{ padding: '20px', borderRadius: '24px', display: 'flex', gap: '20px', alignItems: 'center' }}>
                    <div style={{ position: 'relative', width: '160px', height: '90px', borderRadius: '16px', overflow: 'hidden', flexShrink: 0 }}>
                      <img src={video.thumbnail} alt={video.title} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                      {video.duration && (
                        <span style={{ 
                          position: 'absolute', 
                          bottom: '6px', 
                          right: '6px', 
                          backgroundColor: 'rgba(0,0,0,0.85)', 
                          color: '#fff', 
                          padding: '2px 6px', 
                          borderRadius: '4px', 
                          fontSize: '0.75rem', 
                          fontWeight: 600 
                        }}>
                          {formatDuration(video.duration)}
                        </span>
                      )}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '12px' }}>
                        <h3 style={{ margin: '0 0 12px 0', fontSize: '1.1rem', fontWeight: 600, color: '#1d1d1f', lineHeight: 1.4 }}>
                          {video.title}
                        </h3>
                        <a 
                          href={`https://www.youtube.com/watch?v=${video.id}`} 
                          target="_blank" 
                          rel="noopener noreferrer" 
                          style={{ color: '#007aff', display: 'flex', alignItems: 'center', marginTop: '2px' }}
                          title="Xem trên YouTube"
                        >
                          <ExternalLink size={18} />
                        </a>
                      </div>
                      <div style={{ display: 'flex', gap: '24px', color: '#515154', fontSize: '0.95rem', fontWeight: 500, flexWrap: 'wrap', alignItems: 'center', marginBottom: '12px' }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>👁️ {formatNumber(video.views)} lượt xem</span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>👍 {formatNumber(video.likes)} lượt thích</span>
                        <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>💬 {formatNumber(video.comments)} bình luận</span>
                        {getPrivacyBadge(video.privacy)}
                      </div>
                      <div style={{ fontSize: '0.85rem', color: '#86868b', fontWeight: 500 }}>
                        ID: <code style={{ backgroundColor: 'rgba(0,0,0,0.05)', padding: '2px 6px', borderRadius: '4px' }}>{video.id}</code> • Đã đăng: {new Date(video.published_at).toLocaleDateString("vi-VN")}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
});
