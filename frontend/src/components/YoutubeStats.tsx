import { useEffect, useState } from "react";
import { getYoutubeAuthUrl, getYoutubeStats, sendYoutubeCallbackCode } from "../lib/api";

export function YoutubeStats() {
  const [stats, setStats] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Check if we are returning from OAuth callback
    const urlParams = new URLSearchParams(window.location.search);
    const code = urlParams.get("code");
    
    if (code) {
      setLoading(true);
      sendYoutubeCallbackCode(code)
        .then(() => {
          // Remove code from URL
          window.history.replaceState({}, document.title, window.location.pathname);
          fetchStats();
        })
        .catch(err => {
          setError(err.message || "Failed to authenticate with YouTube");
          setLoading(false);
        });
    } else {
      fetchStats();
    }
  }, []);

  const fetchStats = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getYoutubeStats();
      setStats(data);
    } catch (err: any) {
      setError(err.message || "Lỗi tải dữ liệu. Vui lòng xác thực kênh Youtube.");
    } finally {
      setLoading(false);
    }
  };

  const handleAuth = async () => {
    try {
      const res = await getYoutubeAuthUrl();
      window.location.href = res.url;
    } catch (err: any) {
      setError(err.message || "Failed to get auth URL");
    }
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

        {error && <div style={{ backgroundColor: 'rgba(255, 59, 48, 0.1)', color: '#ff3b30', padding: '16px', borderRadius: '16px', marginBottom: '24px', fontWeight: 500, border: '1px solid rgba(255, 59, 48, 0.2)' }}>{error}</div>}

        {loading ? (
          <div style={{ textAlign: 'center', padding: '40px', color: '#86868b', fontWeight: 500 }}>Đang tải dữ liệu...</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {stats.length === 0 && !error ? (
              <div style={{ textAlign: 'center', padding: '40px', color: '#86868b', fontWeight: 500 }}>Không có video nào được tìm thấy.</div>
            ) : (
              stats.map(video => (
                <div key={video.id} className="yt-stat-item ios-glass" style={{ padding: '20px', borderRadius: '24px', display: 'flex', gap: '20px', alignItems: 'center' }}>
                  <img src={video.thumbnail} alt={video.title} className="yt-stat-img" />
                  <div style={{ flex: 1 }}>
                    <h3 style={{ margin: '0 0 12px 0', fontSize: '1.1rem', fontWeight: 600, color: '#1d1d1f' }}>{video.title}</h3>
                    <div style={{ display: 'flex', gap: '24px', color: '#515154', fontSize: '0.95rem', fontWeight: 500 }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>👁️ {video.views}</span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>👍 {video.likes}</span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>💬 {video.comments}</span>
                    </div>
                    <div style={{ marginTop: '16px', fontSize: '0.85rem', color: '#86868b', fontWeight: 500 }}>
                      ID: {video.id} • Đã đăng: {new Date(video.published_at).toLocaleDateString()}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}
