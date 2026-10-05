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
# 3. Khai báo các landmark của mắt
# ============================================================
# Iris trái
LEFT_IRIS = [474, 475, 476, 477]
# Iris phải
RIGHT_IRIS = [469, 470, 471, 472]
# Khóe mắt trái
LEFT_EYE_CORNERS = [362, 263]
# Khóe mắt phải
RIGHT_EYE_CORNERS = [33, 133]
# Điểm trên và dưới của mắt trái
LEFT_EYE_VERTICAL = [386, 374]
# Điểm trên và dưới của mắt phải
RIGHT_EYE_VERTICAL = [159, 145]
# ============================================================
# 4. Hàm tính vị trí tương đối của iris
# ============================================================
def get_iris_position(
    landmarks,
    iris_indices,
    eye_corners,
    eye_vertical,
    frame_width,
    frame_height
):
    # --------------------------------------------------------
    # Lấy các điểm iris
    # --------------------------------------------------------
    iris_points = []
    for index in iris_indices:
        x = landmarks[index].x * frame_width
        y = landmarks[index].y * frame_height
        iris_points.append((x, y))
    # --------------------------------------------------------
    # Tính tâm của iris
    # --------------------------------------------------------
    iris_center_x = sum(
        point[0] for point in iris_points
    ) / len(iris_points)
    iris_center_y = sum(
        point[1] for point in iris_points
    ) / len(iris_points)
    # --------------------------------------------------------
    # Lấy hai khóe mắt
    # --------------------------------------------------------
    corner_1 = landmarks[eye_corners[0]]
    corner_2 = landmarks[eye_corners[1]]
    corner_1_x = corner_1.x * frame_width
    corner_1_y = corner_1.y * frame_height
    corner_2_x = corner_2.x * frame_width
    corner_2_y = corner_2.y * frame_height
    # --------------------------------------------------------
    # Lấy điểm trên và dưới của mắt
    # --------------------------------------------------------
    top = landmarks[eye_vertical[0]]
    bottom = landmarks[eye_vertical[1]]
    top_x = top.x * frame_width
    top_y = top.y * frame_height
    bottom_x = bottom.x * frame_width
    bottom_y = bottom.y * frame_height
    # --------------------------------------------------------
    # Tính chiều rộng và chiều cao của mắt
    # --------------------------------------------------------
    eye_width = (
        (corner_2_x - corner_1_x) ** 2 +
        (corner_2_y - corner_1_y) ** 2
    ) ** 0.5
    eye_height = (
        (bottom_x - top_x) ** 2 +
        (bottom_y - top_y) ** 2
    ) ** 0.5
    # --------------------------------------------------------
    # Tránh chia cho 0
    # --------------------------------------------------------
    if eye_width == 0 or eye_height == 0:
        return 0.5, 0.5, (
            iris_center_x,
            iris_center_y
        )
    # --------------------------------------------------------
    # Tính vị trí ngang của iris
    # --------------------------------------------------------
    iris_horizontal_ratio = (
        (
            (
                iris_center_x - corner_1_x
            ) ** 2
            +
            (
                iris_center_y - corner_1_y
            ) ** 2
        ) ** 0.5
    ) / eye_width
    # --------------------------------------------------------
    # Tính vị trí dọc của iris
    # --------------------------------------------------------
    iris_vertical_ratio = (
        (
            (
                iris_center_x - top_x
            ) ** 2
            +
            (
                iris_center_y - top_y
            ) ** 2
        ) ** 0.5
    ) / eye_height
    return (
        iris_horizontal_ratio,
        iris_vertical_ratio,
        (iris_center_x, iris_center_y)
    )
# ============================================================
# 5. Xác định hướng nhìn
# ============================================================
def detect_eye_direction(
    horizontal_ratio,
    vertical_ratio
):
    # --------------------------------------------------------
    # Ngưỡng xác định hướng nhìn
    # --------------------------------------------------------
    HORIZONTAL_LEFT_THRESHOLD = 0.58
    HORIZONTAL_RIGHT_THRESHOLD = 0.42
    VERTICAL_UP_THRESHOLD = 0.40
    VERTICAL_DOWN_THRESHOLD = 0.60
    # --------------------------------------------------------
    # Xác định hướng ngang
    # --------------------------------------------------------
    if horizontal_ratio < HORIZONTAL_RIGHT_THRESHOLD:
        horizontal_direction = "RIGHT"
    elif horizontal_ratio > HORIZONTAL_LEFT_THRESHOLD:
        horizontal_direction = "LEFT"
    else:
        horizontal_direction = ""
    # --------------------------------------------------------
    # Xác định hướng dọc
    # --------------------------------------------------------
    if vertical_ratio < VERTICAL_UP_THRESHOLD:
        vertical_direction = "UP"
    elif vertical_ratio > VERTICAL_DOWN_THRESHOLD:
        vertical_direction = "DOWN"
    else:
        vertical_direction = ""
    # --------------------------------------------------------
    # Kết hợp hướng ngang và hướng dọc
    # --------------------------------------------------------
    if horizontal_direction and vertical_direction:
        return (
            horizontal_direction
            + " "
            + vertical_direction
        )
    elif horizontal_direction:
        return horizontal_direction

    elif vertical_direction:
        return vertical_direction
    else:
        return "CENTER"
# ============================================================
# 6. Mở webcam và thực hiện Eye Tracking
# ============================================================
cap = cv2.VideoCapture(0)
start_time = time.time()
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
        # Lấy kích thước frame
        # ----------------------------------------------------
        frame_height, frame_width, _ = frame.shape
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
            (time.time() - start_time) * 1000
        )
        # ----------------------------------------------------
        # Chạy Face Landmarker
        # ----------------------------------------------------
        detection_result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )
        # ====================================================
        # 7. Xử lý kết quả
        # ====================================================
        if detection_result.face_landmarks:
            for face_landmarks in detection_result.face_landmarks:
                # ------------------------------------------------
                # Tính vị trí iris mắt trái
                # ------------------------------------------------
                left_horizontal, left_vertical, left_center = (
                    get_iris_position(
                        face_landmarks,
                        LEFT_IRIS,
                        LEFT_EYE_CORNERS,
                        LEFT_EYE_VERTICAL,
                        frame_width,
                        frame_height
                    )
                )
                # ------------------------------------------------
                # Tính vị trí iris mắt phải
                # ------------------------------------------------
                right_horizontal, right_vertical, right_center = (
                    get_iris_position(
                        face_landmarks,
                        RIGHT_IRIS,
                        RIGHT_EYE_CORNERS,
                        RIGHT_EYE_VERTICAL,
                        frame_width,
                        frame_height
                    )
                )
                # ------------------------------------------------
                # Lấy trung bình hai mắt
                # ------------------------------------------------
                horizontal_ratio = (
                    left_horizontal + right_horizontal
                ) / 2
                vertical_ratio = (
                    left_vertical + right_vertical
                ) / 2
                # ------------------------------------------------
                # Xác định hướng mắt
                # ------------------------------------------------
                eye_direction = detect_eye_direction(
                    horizontal_ratio,
                    vertical_ratio
                )
                # ------------------------------------------------
                # Vẽ tâm iris mắt trái
                # ------------------------------------------------
                cv2.circle(
                    frame,
                    (
                        int(left_center[0]),
                        int(left_center[1])
                    ),
                    4,
                    (0, 255, 0),
                    -1
                )
                # ------------------------------------------------
                # Vẽ tâm iris mắt phải
                # ------------------------------------------------
                cv2.circle(
                    frame,
                    (
                        int(right_center[0]),
                        int(right_center[1])
                    ),
                    4,
                    (0, 255, 0),
                    -1
                )
                # ------------------------------------------------
                # Vẽ các landmark của iris
                # ------------------------------------------------
                for index in LEFT_IRIS + RIGHT_IRIS:

                    landmark = face_landmarks[index]

                    x = int(
                        landmark.x * frame_width
                    )

                    y = int(
                        landmark.y * frame_height
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        2,
                        (0, 255, 0),
                        -1
                    )
                # ------------------------------------------------
                # Hiển thị hướng nhìn
                # ------------------------------------------------
                cv2.putText(
                    frame,
                    f"EYE TRACKING: {eye_direction}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )
                # ------------------------------------------------
                # Hiển thị giá trị tracking
                # ------------------------------------------------
                cv2.putText(
                    frame,
                    f"H: {horizontal_ratio:.2f}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    f"V: {vertical_ratio:.2f}",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
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
        # HIỂN THỊ CAMERA
        # ====================================================
        cv2.imshow(
            'MediaPipe Tasks API - Eye Tracking',
            frame
        )
        # ----------------------------------------------------
        # Nhấn ESC để thoát
        # ----------------------------------------------------
        if cv2.waitKey(5) & 0xFF == 27:
            break
# ============================================================
# 8. Giải phóng tài nguyên
# ============================================================
cap.release()
cv2.destroyAllWindows()