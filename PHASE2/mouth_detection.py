import cv2
import mediapipe as mp
import time
import math
# ============================================================
# 1. Khởi tạo các lớp từ MediaPipe Tasks API
# ============================================================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
# ============================================================
# 2. Cấu hình Face Landmarker
# ============================================================
options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path='face_landmarker.task'
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)
# ============================================================
# 3. Hàm tính khoảng cách giữa 2 điểm
# ============================================================
def distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )
# ============================================================
# 4. Biến dùng để phát hiện ngáp
# ============================================================
# Thời điểm bắt đầu mở miệng lớn
yawn_start_time = None
# Thời điểm kết thúc hiển thị cảnh báo
message_until = 0
# Ngưỡng độ mở miệng
MOUTH_OPEN_THRESHOLD = 0.75
# Miệng phải mở ít nhất khoảng thời gian này mới được coi là ngáp
YAWN_DURATION = 0.4
# ============================================================
# 5. Mở webcam
# ============================================================
cap = cv2.VideoCapture(0)
# ============================================================
# 6. Khởi tạo Face Landmarker
# ============================================================
with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        # ----------------------------------------------------
        # Đọc frame từ webcam
        # ----------------------------------------------------
        success, frame = cap.read()
        if not success:
            print("Không thể đọc từ camera.")
            break
        # Lật ảnh giống gương
        frame = cv2.flip(frame, 1)
        # BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
        # ----------------------------------------------------
        # Chuyển thành MediaPipe Image
        # ----------------------------------------------------
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )
        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------
        timestamp_ms = int(time.time() * 1000)
        # ----------------------------------------------------
        # MediaPipe phát hiện khuôn mặt
        # ----------------------------------------------------
        detection_result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # ====================================================
        # 7. Xử lý landmark
        # ====================================================
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                # ------------------------------------------------
                # Lấy các landmark quanh miệng
                # ------------------------------------------------
                # Môi trên
                upper_lip = face_landmarks[13]
                # Môi dưới
                lower_lip = face_landmarks[14]
                # Khóe miệng trái
                left_mouth = face_landmarks[61]
                # Khóe miệng phải
                right_mouth = face_landmarks[291]
                # ------------------------------------------------
                # Tính khoảng cách dọc miệng
                # ------------------------------------------------
                vertical_distance = distance(
                    upper_lip,
                    lower_lip
                )
                # ------------------------------------------------
                # Tính khoảng cách ngang miệng
                # ------------------------------------------------
                horizontal_distance = distance(
                    left_mouth,
                    right_mouth
                )
                # ------------------------------------------------
                # Tính tỷ lệ mở miệng
                # ------------------------------------------------
                if horizontal_distance > 0:
                    mouth_ratio = (
                        vertical_distance /
                        horizontal_distance
                    )
                else:
                    mouth_ratio = 0
                # =================================================
                # 8. Kiểm tra miệng có đang mở lớn không
                # =================================================
                current_time = time.time()
                if mouth_ratio > MOUTH_OPEN_THRESHOLD:
                    # Nếu đây là thời điểm bắt đầu mở miệng
                    if yawn_start_time is None:
                        yawn_start_time = current_time
                    # Kiểm tra miệng đã mở đủ lâu chưa
                    elif current_time - yawn_start_time >= YAWN_DURATION:
                        # Kích hoạt cảnh báo 2 giây
                        message_until = current_time + 2
                        # Reset để tránh kích hoạt liên tục
                        yawn_start_time = None
                else:
                    # Miệng đã đóng
                    yawn_start_time = None
                # =================================================
                # 9. Vẽ landmarks
                # =================================================

                for landmark in face_landmarks:

                    x = int(
                        landmark.x *
                        frame.shape[1]
                    )

                    y = int(
                        landmark.y *
                        frame.shape[0]
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )


                # =================================================
                # 10. Hiển thị thông báo
                # =================================================

                if current_time < message_until:

                    cv2.putText(
                        frame,
                        "CANH BAO: BAN DANG NGAP!",
                        (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        3
                    )
                # ------------------------------------------------
                # Hiển thị mouth ratio để debug
                # ------------------------------------------------
                cv2.putText(
                    frame,
                    f"Mouth: {mouth_ratio:.2f}",
                    (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
        # ====================================================
        # 11. Hiển thị webcam
        # ====================================================
        cv2.imshow(
            'MediaPipe Face Mesh - Yawn Detection',
            frame
        )
        # ====================================================
        # 12. ESC để thoát
        # ====================================================
        if cv2.waitKey(5) & 0xFF == 27:
            break
# ============================================================
# 13. Giải phóng tài nguyên
# ============================================================
cap.release()
cv2.destroyAllWindows()