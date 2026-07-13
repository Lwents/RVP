# Review phim đa phương thức

Luồng Review phim không còn dựng trực tiếp từ transcript. Mỗi job chạy theo thứ tự:

1. ASR có timestamp; nếu dịch tạm bị lỗi/quota, Gemini phân tích trực tiếp ngôn ngữ nguồn.
2. Phát hiện shot trên toàn video và lấy tối thiểu ba keyframe cho mỗi scene.
3. Gemini Pro phân tích hình, thoại và hard-sub/OCR thành `scene_timeline.json`.
4. Tạo `event_timeline.json` đúng source time, chỉ đánh dấu `verified` khi có evidence.
5. Viết từng câu review gắn `event_id`, `required_visuals` và scene thật.
6. Chấm visual-semantic bằng nhiều keyframe, tạo top 3 ứng viên và `edit_decision_list.json`.
7. Chặn render khi QA dưới 90/100 hoặc bất kỳ câu nào khớp cảnh dưới 0,75.
8. Render theo đúng duration voice/EDL, sau đó Gemini kiểm tra ba frame trong mỗi cửa sổ voice.
9. Cảnh lỗi được thử phương án thay thế đủ điểm; không tạo `review_final.mp4` nếu post-render QA vẫn không đạt.

## Chạy

Từ thư mục gốc:

```powershell
npm run dev
```

Lệnh này mở đồng thời 9Router (`20128`), backend (`8000`) và frontend (`5173`). Trang Review phim dùng địa chỉ:

```text
http://localhost:5173/#review
```

## Artifact mỗi job

Trong `backend/storage/review_jobs/<job_id>/`:

- `scene_timeline.json`
- `event_timeline.json`
- `verified_review_script.json`
- `edit_decision_list.json`
- `quality_report.json`
- `review_draft.mp4`
- `quality_report_post_render.json`
- `review_final.mp4` (chỉ có khi QA đạt)

Trên giao diện, mỗi câu có thumbnail, timestamp nguồn/voice, điểm/lý do khớp, ba cảnh thay thế và ô sửa lời. Câu dưới 75% được đánh đỏ. Lời sửa được Gemini kiểm chứng lại với event và keyframe trước khi lưu.

## Test regression

```powershell
cd backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Regression kiểm tra đặc biệt lỗi cũ: voice 156,5 giây không còn bị kéo thành video 480 giây, subtitle dùng đúng voice range và clip không chạy ra ngoài scene evidence.
