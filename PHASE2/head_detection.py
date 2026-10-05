import cv2
import mediapipe as mp
import time
import math
# ============================================================
# 1. Khởi tạo MediaPipe
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
# 3. Hàm tính khoảng cách
# ============================================================
def distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )
# ============================================================
# 4. Hàm phát hiện hướng đầu
# ============================================================
def detect_head_direction(face_landmarks):
    # --------------------------------------------------------
    # Lấy các landmark quan trọng
    # --------------------------------------------------------
    nose = face_landmarks[1]
    left_eye = face_landmarks[33]
    right_eye = face_landmarks[263]
    forehead = face_landmarks[10]
    chin = face_landmarks[152]
    # ========================================================
    # A. XÁC ĐỊNH TRÁI / PHẢI
    # ========================================================
    # Trung tâm hai mắt
    eye_center_x = (
        left_eye.x +
        right_eye.x
    ) / 2
    # Khoảng cách hai mắt
    eye_distance = distance(
        left_eye,
        right_eye
    )
    # Tránh chia cho 0
    if eye_distance == 0:
        return "UNKNOWN"
    # --------------------------------------------------------
    # Mũi lệch bao nhiêu so với trung tâm hai mắt
    # --------------------------------------------------------
    horizontal_ratio = (
        nose.x - eye_center_x
    ) / eye_distance
    # ========================================================
    # B. XÁC ĐỊNH LÊN / XUỐNG
    # ========================================================
    # Trung tâm theo chiều dọc của khuôn mặt
    face_center_y = (
        forehead.y +
        chin.y
    ) / 2
    # Khoảng cách chiều dọc khuôn mặt
    face_height = abs(
        chin.y -
        forehead.y
    )
    if face_height == 0:
        return "UNKNOWN"
    # Mũi lệch bao nhiêu theo chiều dọc
    vertical_ratio = (
        nose.y -
        face_center_y
    ) / face_height
    # ========================================================
    # C. ĐẶT NGƯỠNG
    # ========================================================
    HORIZONTAL_THRESHOLD = 0.20
    VERTICAL_THRESHOLD = 0.12
    # ========================================================
    # D. PHÂN LOẠI
    # ========================================================
    # ----- QUAY TRÁI / PHẢI -----
    if horizontal_ratio < -HORIZONTAL_THRESHOLD:
        return "LEFT"
    elif horizontal_ratio > HORIZONTAL_THRESHOLD:
        return "RIGHT"
    # ----- NGẨNG / CÚI -----
    elif vertical_ratio < -VERTICAL_THRESHOLD:
        return "UP"
    elif vertical_ratio > VERTICAL_THRESHOLD:
        return "DOWN"
    # ----- NHÌN THẲNG -----
    else:
        return "FORWARD"
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
        # Đọc frame
        # ----------------------------------------------------
        success, frame = cap.read()
        if not success:
            print("Không thể đọc từ camera.")
            break
        # ----------------------------------------------------
        # Lật ảnh
        # ----------------------------------------------------
        frame = cv2.flip(frame, 1)
        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------
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
        timestamp_ms = int(
            time.time() * 1000
        )
        # ----------------------------------------------------
        # Nhận diện khuôn mặt
        # ----------------------------------------------------
        detection_result = (
            landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )
        )
        # ====================================================
        # 7. Xử lý landmark
        # ====================================================
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                # =================================================
                # Xác định hướng đầu
                # =================================================

                head_direction = detect_head_direction(
                    face_landmarks
                )
                # =================================================
                # Vẽ toàn bộ landmark
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
                # Hiển thị hướng đầu
                # =================================================
                if head_direction == "FORWARD":
                    text = "NHIN THANG"
                elif head_direction == "LEFT":
                    text = "QUAY TRAI"
                elif head_direction == "RIGHT":
                    text = "QUAY PHAI"
                elif head_direction == "UP":
                    text = "NGANG LEN"
                elif head_direction == "DOWN":
                    text = "CUI XUONG"
                else:
                    text = "UNKNOWN"
                # =================================================
                # Vẽ text
                # =================================================
                cv2.putText(
                    frame,
                    f"HEAD: {text}",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    3
                )
        # ====================================================
        # 8. Hiển thị webcam
        # ====================================================
        cv2.imshow(
            "MediaPipe Head Detection",
            frame
        )
        # ====================================================
        # 9. ESC để thoát
        # ====================================================
        if cv2.waitKey(5) & 0xFF == 27:
            break
# ============================================================
# 10. Giải phóng
# ============================================================
cap.release()
cv2.destroyAllWindows()