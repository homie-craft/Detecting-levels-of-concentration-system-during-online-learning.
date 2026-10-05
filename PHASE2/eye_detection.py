import cv2
import mediapipe as mp
import time
# ============================================================
# 1. KHỞI TẠO MEDIAPIPE FACE LANDMARKER
# ============================================================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path='face_landmarker.task'
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)
# ============================================================
# 2. CÁC HÀM TÍNH TOÁN
# ============================================================
def distance(p1, p2):
    return (
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    ) ** 0.5
# ============================================================
# 3. TÍNH EAR - EYE ASPECT RATIO
# ============================================================
def calculate_ear(face_landmarks, eye_indices):
    p1 = face_landmarks[eye_indices[0]]
    p2 = face_landmarks[eye_indices[1]]
    p3 = face_landmarks[eye_indices[2]]
    p4 = face_landmarks[eye_indices[3]]
    p5 = face_landmarks[eye_indices[4]]
    p6 = face_landmarks[eye_indices[5]]

    # Khoảng cách theo chiều dọc
    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)

    # Khoảng cách theo chiều ngang
    horizontal = distance(p1, p4)

    # Công thức EAR
    ear = (
        vertical_1 + vertical_2
    ) / (
        2.0 * horizontal
    )

    return ear
# ============================================================
# 4. DETECT EYE
# ============================================================
def detect_eye(face_landmarks):
    # Landmark của mắt trái
    LEFT_EYE = [
        33,
        160,
        158,
        133,
        153,
        144
    ]
    # Landmark của mắt phải
    RIGHT_EYE = [
        362,
        385,
        387,
        263,
        373,
        380
    ]
    # Tính EAR cho từng mắt
    left_ear = calculate_ear(
        face_landmarks,
        LEFT_EYE
    )
    right_ear = calculate_ear(
        face_landmarks,
        RIGHT_EYE
    )
    # EAR trung bình
    ear = (left_ear + right_ear) / 2
    # Ngưỡng phát hiện mắt nhắm
    EAR_THRESHOLD = 0.20

    if ear < EAR_THRESHOLD:
        status = "NHAM MAT"
    else:
        status = "MO MAT"

    return status, ear
# ============================================================
# 5. MỞ WEBCAM
# ============================================================
cap = cv2.VideoCapture(0)
# ============================================================
# 6. CHẠY FACE LANDMARKER
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
        # Chuyển thành mp.Image
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
        # Face Landmarker
        # ----------------------------------------------------
        detection_result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )
        # ====================================================
        # 7. XỬ LÝ FACE LANDMARKS
        # ====================================================
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                # --------------------------------------------
                # Detect mắt
                # --------------------------------------------
                eye_status, ear = detect_eye(
                    face_landmarks
                )
                # --------------------------------------------
                # Vẽ 468 landmarks
                # --------------------------------------------
                for landmark in face_landmarks:
                    x = int(
                        landmark.x * frame.shape[1]
                    )

                    y = int(
                        landmark.y * frame.shape[0]
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )
                # --------------------------------------------
                # Hiển thị trạng thái mắt
                # --------------------------------------------
                cv2.putText(
                    frame,
                    f"MAT: {eye_status}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                # --------------------------------------------
                # Hiển thị EAR
                # --------------------------------------------
                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}",
                    (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )
        # ====================================================
        # 8. HIỂN THỊ CAMERA
        # ====================================================
        cv2.imshow(
            "Eye Detection",
            frame
        )
        # ESC để thoát
        if cv2.waitKey(5) & 0xFF == 27:
            break
# ============================================================
# 9. GIẢI PHÓNG CAMERA
# ============================================================
cap.release()
cv2.destroyAllWindows()