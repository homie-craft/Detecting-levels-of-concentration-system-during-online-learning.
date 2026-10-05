import cv2
import mediapipe as mp
import time
# ============================================================
# 1. Khởi tạo các lớp từ Tasks API
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
# 3. Mở webcam
# ============================================================
cap = cv2.VideoCapture(0)
# ============================================================
# 4. Khởi tạo Face Landmarker
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
        # ----------------------------------------------------
        # Lật ảnh giống gương
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
        # Chuyển sang mp.Image
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
        # Chạy Face Landmarker
        # ----------------------------------------------------
        detection_result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )
        # ====================================================
        # 5. KIỂM TRA CÓ NHÌN RÕ KHUÔN MẶT HAY KHÔNG
        # ====================================================
        if detection_result.face_landmarks:
            # ------------------------------------------------
            # Có khuôn mặt
            # ------------------------------------------------
            for face_landmarks in detection_result.face_landmarks:
                # --------------------------------------------
                # Vẽ 468 landmarks
                # --------------------------------------------
                for landmark in face_landmarks:
                    # Chuyển tọa độ chuẩn hóa
                    # sang tọa độ pixel
                    x = int(
                        landmark.x * frame.shape[1]
                    )
                    y = int(
                        landmark.y * frame.shape[0]
                    )
                    # Vẽ điểm
                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )
        else:
            # =================================================
            # KHÔNG PHÁT HIỆN ĐƯỢC KHUÔN MẶT
            # =================================================
            cv2.putText(
                frame,
                "KHONG NHIN RO MAT",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )
        # ====================================================
        # 6. HIỂN THỊ CAMERA
        # ====================================================
        cv2.imshow(
            'MediaPipe Tasks API - Face Mesh',
            frame
        )
        # ----------------------------------------------------
        # Nhấn ESC để thoát
        # ----------------------------------------------------
        if cv2.waitKey(5) & 0xFF == 27:
            break
# ============================================================
# 7. Giải phóng tài nguyên
# ============================================================
cap.release()
cv2.destroyAllWindows()