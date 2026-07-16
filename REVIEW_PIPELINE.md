# Review phim đa phương thức

Luồng Review phim không còn dựng trực tiếp từ transcript. Mỗi job chạy theo thứ tự:

1. ASR có timestamp; nếu dịch tạm bị lỗi/quota, Gemini phân tích trực tiếp ngôn ngữ nguồn.
2. Phát hiện shot trên toàn video và lấy tối thiểu ba keyframe cho mỗi scene.
3. Model Gemini đa phương thức được cấu hình tập trung trong `backend/app/core.py` phân tích hình, thoại và hard-sub/OCR thành `scene_timeline.json`.
4. Tạo `event_timeline.json` đúng source time, chỉ đánh dấu `verified` khi có evidence; mỗi khoảng thời gian của phim phải có mốc sự kiện để tránh dồn hết nội dung vào một hồi rồi nhảy cóc.
5. Tính ngân sách lời đọc từ số phút người dùng chọn. Tốc độ chuẩn hiện tại là khoảng 195 từ/phút, cho phép lệch tối đa 10%.
6. Viết từng câu review gắn `event_id`, vai trò `hook/context/conflict/climax/resolution`, `required_visuals` và scene thật. Mọi event đã kiểm chứng phải xuất hiện đúng thứ tự.
7. Gemini chấm lại độ liền mạch, quan hệ chuyển đoạn và mức bám phong cách đã chọn (`Kể chuyện`, `Nhanh gọn`, `Cảm xúc`, `Duyên hài`). Nếu chưa đạt, kịch bản được yêu cầu viết lại tối đa hai lần theo đúng phản hồi và vẫn bị chặn QA nếu còn lỗi.
8. Chấm visual-semantic bằng nhiều keyframe, tạo top 3 ứng viên và `edit_decision_list.json`. Timestamp cảnh được chọn không được đi lùi so với câu trước.
9. Chặn render khi QA dưới 90/100, sai thời lượng, thiếu mạch truyện/phong cách, thiếu phủ timeline hoặc bất kỳ câu nào khớp cảnh dưới 0,75.
10. Tạo giọng đọc rồi đo thời lượng file thật. Chỉ khi thời lượng nằm trong ±10% mục tiêu mới render theo đúng duration voice/EDL; tuyệt đối không kéo một voice 1 phút thành video 8 phút.
11. Sau render, Gemini kiểm tra ba frame trong mỗi cửa sổ voice. Cảnh lỗi được thử phương án thay thế đủ điểm; không tạo `review_final.mp4` nếu post-render QA vẫn không đạt.

## Hợp đồng thời lượng và phong cách

- Trường `Thời lượng review` nhận 1–30 phút và là mục tiêu bắt buộc, không phải gợi ý hay giới hạn tối đa.
- Ví dụ 8 phút cần khoảng 1.560 từ, hợp lệ trong khoảng 1.404–1.716 từ; giọng đọc thật phải dài 432–528 giây.
- `Kể chuyện`: ưu tiên mạch nhân quả và chuyển đoạn mềm.
- `Nhanh gọn`: câu ngắn, động từ mạnh nhưng vẫn giữ nguyên nhân–xung đột–kết quả.
- `Cảm xúc`: nhấn biểu cảm/lựa chọn có evidence, không tự suy diễn nội tâm.
- `Duyên hài`: lấy sự hài từ tình huống thật, không bịa thoại hay phá logic nhân vật.
- Giao diện luôn hiện riêng mục tiêu, thời lượng voice dự kiến, thời lượng video thật và điểm QA về thời lượng, mạch truyện, phong cách, độ phủ timeline.

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

Regression kiểm tra đặc biệt các lỗi cũ: yêu cầu 8 phút không thể xuất kịch bản 1 phút; voice ngắn không bị kéo giả thành 480 giây; scene không đi ngược source time; event phủ toàn bộ các hồi phim; subtitle dùng đúng voice range và clip không chạy ra ngoài scene evidence.
