# MỤC ĐÍCH CỦA CHƯƠNG TRÌNH:
Phát hiện khuôn mặt từ webcam và lấy ra các landmark (điểm đặc trưng).   
Các landmark này sẽ trở thành dữ liệu đầu vào (tọa độ x, y, z) cho các module phát hiện hành vi khác như quay đầu, nhắm mắt, mở miệng.   
# CÁC THƯ VIỆN VÀ CẤU HÌNH SỬ DỤNG:
- cv2: Mở webcam, lật ảnh, xử lý hệ màu BGR và vẽ landmark.   
- mediapipe: Cung cấp API mô hình Face Landmarker.   
- time: Cung cấp timestamp cho các frame.   
- Model sử dụng: face_landmarker.task.   
- Chế độ xử lý: VisionRunningMode.VIDEO.   
- Cấu hình số khuôn mặt: num_faces=1 (chỉ xử lý tối đa một khuôn mặt).  
# CÁC BIẾN SỬ DỤNG:
- frame: Ảnh gốc đọc trực tiếp từ webcam (OpenCV đọc theo thứ tự BGR).   
- rgb_frame: Frame đã chuyển đổi sang hệ màu RGB (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).   
- mp_image: Dữ liệu ảnh đã được ép kiểu sang định dạng của MediaPipe (mp.Image).   
- timestamp_ms: Dấu thời gian của frame tính bằng mili-giây (int(time.time() * 1000)).   
- detection_result: Chứa kết quả thông tin khuôn mặt sau khi model phân tích.   - face_landmarks: Danh sách tọa độ các điểm đặc trưng. Mỗi landmark chứa biến x, y, z đã được chuẩn hóa trong miền giá trị [0;1].   
# CÔNG THỨC CHUYỂN ĐỔI TỌA ĐỘ PIXEL:
- Do landmark.x và landmark.y chỉ là tỷ lệ chuẩn hóa (từ 0 đến 1), ta cần chuyển đổi sang điểm ảnh (pixel) thực tế để vẽ:   
+ Tọa độ X (chiều ngang): x = int(landmark.x * frame.shape[1]) (với frame.shape[1] là chiều rộng ảnh).   
+ Tọa độ Y (chiều dọc): y = int(landmark.y * frame.shape[0]) (với frame.shape[0] là chiều cao ảnh).   
- ĐIỀU KIỆN KIỂM TRA KHUÔN MẶT: 
+ Đặt điều kiện có nhìn rõ khuôn mặt hay không: 
if detection_result.face_landmarks:   
Nếu có khuôn mặt:
    Vòng lặp sẽ duyệt qua các điểm trong face_landmarks.   
    Tính toán tọa độ pixel x, y theo công thức trên.   
    Sử dụng cv2.circle(frame, (x, y), 1, (0, 255, 0), -1) để vẽ các điểm landmark lên màn hình.   
Nếu không phát hiện được khuôn mặt:
    else:   Sử dụng cv2.putText để in cảnh báo "KHONG NHIN RO MAT" lên góc màn hìn