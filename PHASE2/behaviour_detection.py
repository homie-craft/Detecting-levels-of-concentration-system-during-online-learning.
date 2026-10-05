import cv2
import mediapipe as mp
import time
import math
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
# 2. CÁC LANDMARK QUAN TRỌNG
# ============================================================
# -----------------------------
# Mắt trái
# -----------------------------
LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]
# -----------------------------
# Mắt phải
# -----------------------------
RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]
# -----------------------------
# Miệng
# -----------------------------
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
# -----------------------------
# Head direction
# -----------------------------
NOSE = 1
FOREHEAD = 10
CHIN = 152
LEFT_EYE_CENTER = 33
RIGHT_EYE_CENTER = 263
# ============================================================
# 3. CÁC NGƯỠNG
# ============================================================
# -----------------------------
# Eye detection
# -----------------------------
EAR_THRESHOLD = 0.20
# Nhắm mắt >= 5 giây
EYE_CLOSED_TIME = 5.0
# -----------------------------
# Head detection
# -----------------------------
# Sai lệch ngang
HEAD_HORIZONTAL_THRESHOLD = 0.20
# Sai lệch dọc
HEAD_VERTICAL_THRESHOLD = 0.15
# -----------------------------
# Yawn detection
# -----------------------------
MOUTH_OPEN_THRESHOLD = 0.30
# Miệng mở liên tục ít nhất 0.5 giây
YAWN_DURATION = 0.5
# -----------------------------
# Face absence
# -----------------------------
# Không thấy mặt >= 10 giây
FACE_ABSENCE_TIME = 10.0
# -----------------------------
# Behavior counting
# -----------------------------
# Cúi đầu:
# ít nhất 2 lần trong 60 giây
HEAD_DOWN_WINDOW = 60.0
HEAD_DOWN_REQUIRED = 2
# Ngáp:
# ít nhất 2 lần trong 10 phút
YAWN_WINDOW = 600.0
YAWN_REQUIRED = 2
# ============================================================
# 4. CÁC BIẾN TRẠNG THÁI
# ============================================================
# -----------------------------
# Mắt
# -----------------------------
eye_closed_start = None
# -----------------------------
# Cúi đầu
# -----------------------------
head_down_start = None
head_down_active = False
head_down_events = []
# -----------------------------
# Ngáp
# -----------------------------
yawn_start = None
yawn_active = False
yawn_events = []
# -----------------------------
# Không thấy mặt
# -----------------------------
face_missing_start = None
# ============================================================
# 5. HÀM TÍNH KHOẢNG CÁCH
# ============================================================
def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )
# ============================================================
# 6. TÍNH EAR
# ============================================================
def calculate_ear(face_landmarks, eye_indices):
    p1 = face_landmarks[eye_indices[0]]
    p2 = face_landmarks[eye_indices[1]]
    p3 = face_landmarks[eye_indices[2]]
    p4 = face_landmarks[eye_indices[3]]
    p5 = face_landmarks[eye_indices[4]]
    p6 = face_landmarks[eye_indices[5]]
    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    horizontal = distance(p1, p4)
    if horizontal == 0:
        return 0
    ear = (
        vertical_1 + vertical_2
    ) / (
        2 * horizontal
    )

    return ear


# ============================================================
# 7. EYE DETECTION
# ============================================================

def detect_eye(face_landmarks):

    left_ear = calculate_ear(
        face_landmarks,
        LEFT_EYE
    )

    right_ear = calculate_ear(
        face_landmarks,
        RIGHT_EYE
    )

    ear = (
        left_ear +
        right_ear
    ) / 2

    if ear < EAR_THRESHOLD:

        return "NHAM MAT", ear

    else:

        return "MO MAT", ear


# ============================================================
# 8. HEAD DIRECTION
# ============================================================

def detect_head_direction(face_landmarks):

    nose = face_landmarks[NOSE]

    left_eye = face_landmarks[LEFT_EYE_CENTER]
    right_eye = face_landmarks[RIGHT_EYE_CENTER]

    forehead = face_landmarks[FOREHEAD]
    chin = face_landmarks[CHIN]


    # --------------------------------------------------------
    # Tâm hai mắt
    # --------------------------------------------------------

    eye_center_x = (
        left_eye.x +
        right_eye.x
    ) / 2


    eye_distance = distance(
        left_eye,
        right_eye
    )


    if eye_distance == 0:

        return "UNKNOWN", 0, 0


    # --------------------------------------------------------
    # Độ lệch ngang
    # --------------------------------------------------------

    horizontal_ratio = (
        nose.x - eye_center_x
    ) / eye_distance


    # --------------------------------------------------------
    # Trung tâm khuôn mặt theo chiều dọc
    # --------------------------------------------------------

    face_center_y = (
        forehead.y +
        chin.y
    ) / 2


    face_height = abs(
        chin.y -
        forehead.y
    )


    if face_height == 0:

        return "UNKNOWN", horizontal_ratio, 0


    # --------------------------------------------------------
    # Độ lệch dọc
    # --------------------------------------------------------

    vertical_ratio = (
        nose.y -
        face_center_y
    ) / face_height


    # ========================================================
    # XÁC ĐỊNH HƯỚNG
    # ========================================================

    # Quay trái/phải
    if horizontal_ratio < -HEAD_HORIZONTAL_THRESHOLD:

        direction = "NHIN SANG TRAI"

    elif horizontal_ratio > HEAD_HORIZONTAL_THRESHOLD:

        direction = "NHIN SANG PHAI"


    # Cúi/ngẩng
    elif vertical_ratio > HEAD_VERTICAL_THRESHOLD:

        direction = "CUI DAU"

    elif vertical_ratio < -HEAD_VERTICAL_THRESHOLD:

        direction = "NGANG LEN"


    # Nhìn thẳng
    else:

        direction = "NHIN THANG"


    return (
        direction,
        horizontal_ratio,
        vertical_ratio
    )


# ============================================================
# 9. MOUTH RATIO
# ============================================================

def calculate_mouth_ratio(face_landmarks):

    mouth_left = face_landmarks[MOUTH_LEFT]
    mouth_right = face_landmarks[MOUTH_RIGHT]

    mouth_top = face_landmarks[MOUTH_TOP]
    mouth_bottom = face_landmarks[MOUTH_BOTTOM]


    vertical_distance = distance(
        mouth_top,
        mouth_bottom
    )

    horizontal_distance = distance(
        mouth_left,
        mouth_right
    )


    if horizontal_distance == 0:

        return 0


    ratio = (
        vertical_distance /
        horizontal_distance
    )

    return ratio


# ============================================================
# 10. KIỂM TRA NGÁP
# ============================================================

def detect_yawn(face_landmarks, current_time):

    global yawn_start
    global yawn_active


    mouth_ratio = calculate_mouth_ratio(
        face_landmarks
    )


    # --------------------------------------------------------
    # Miệng đang mở
    # --------------------------------------------------------

    if mouth_ratio >= MOUTH_OPEN_THRESHOLD:

        if yawn_start is None:

            yawn_start = current_time

        duration = (
            current_time -
            yawn_start
        )


        # ----------------------------------------------------
        # Miệng mở đủ lâu
        # ----------------------------------------------------

        if (
            duration >= YAWN_DURATION
            and not yawn_active
        ):

            yawn_active = True

            return True, mouth_ratio


    # --------------------------------------------------------
    # Miệng đóng lại
    # --------------------------------------------------------

    else:

        yawn_start = None
        yawn_active = False


    return False, mouth_ratio


# ============================================================
# 11. XÓA CÁC EVENT CŨ
# ============================================================

def clean_old_events(events, current_time, window):

    events[:] = [
        event_time
        for event_time in events
        if current_time - event_time <= window
    ]


# ============================================================
# 12. MAIN PROGRAM
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("Không thể mở webcam.")

    exit()


with FaceLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        # ====================================================
        # ĐỌC FRAME
        # ====================================================

        success, frame = cap.read()

        if not success:

            print("Không thể đọc từ camera.")

            break


        # ====================================================
        # LẬT ẢNH GIỐNG CAMERA SELFIE
        # ====================================================

        frame = cv2.flip(frame, 1)


        # ====================================================
        # BGR → RGB
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # TẠO MP IMAGE
        # ====================================================

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ====================================================
        # TIMESTAMP
        # ====================================================

        timestamp_ms = int(
            time.monotonic() * 1000
        )


        # ====================================================
        # FACE LANDMARKER
        # ====================================================

        detection_result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # Thời gian hiện tại
        current_time = time.monotonic()


        # ====================================================
        # 13. CÓ KHUÔN MẶT
        # ====================================================

        if detection_result.face_landmarks:

            # ------------------------------------------------
            # Reset bộ đếm không thấy mặt
            # ------------------------------------------------

            face_missing_start = None


            for face_landmarks in detection_result.face_landmarks:

                # ====================================================
                # EYE DETECTION
                # ====================================================

                eye_status, ear = detect_eye(
                    face_landmarks
                )


                # ----------------------------------------------------
                # Kiểm tra nhắm mắt lâu
                # ----------------------------------------------------

                if eye_status == "NHAM MAT":

                    if eye_closed_start is None:

                        eye_closed_start = current_time


                    closed_duration = (
                        current_time -
                        eye_closed_start
                    )

                else:

                    eye_closed_start = None

                    closed_duration = 0


                # ====================================================
                # HEAD DETECTION
                # ====================================================

                (
                    head_direction,
                    horizontal_ratio,
                    vertical_ratio
                ) = detect_head_direction(
                    face_landmarks
                )


                # ====================================================
                # CÚI ĐẦU
                # ====================================================

                if head_direction == "CUI DAU":

                    # --------------------------------------------
                    # Bắt đầu một lần cúi mới
                    # --------------------------------------------

                    if not head_down_active:

                        head_down_active = True

                        head_down_start = current_time

                else:

                    # --------------------------------------------
                    # Kết thúc lần cúi
                    # --------------------------------------------

                    if head_down_active:

                        head_down_events.append(
                            current_time
                        )

                    head_down_active = False
                    head_down_start = None


                # ------------------------------------------------
                # Xóa những lần cúi quá 60 giây
                # ------------------------------------------------

                clean_old_events(
                    head_down_events,
                    current_time,
                    HEAD_DOWN_WINDOW
                )


                # ====================================================
                # YAWN DETECTION
                # ====================================================

                is_yawn, mouth_ratio = detect_yawn(
                    face_landmarks,
                    current_time
                )


                # ------------------------------------------------
                # Nếu vừa phát hiện một lần ngáp
                # ------------------------------------------------

                if is_yawn:

                    yawn_events.append(
                        current_time
                    )


                # ------------------------------------------------
                # Xóa những lần ngáp quá 10 phút
                # ------------------------------------------------

                clean_old_events(
                    yawn_events,
                    current_time,
                    YAWN_WINDOW
                )


                # ====================================================
                # 14. VẼ LANDMARKS
                # ====================================================

                for landmark in face_landmarks:

                    x = int(
                        landmark.x *
                        frame.shape[1]
                    )

                    y = int(
                        landmark.y *
                        frame.shape[0]
                    )


                    if (
                        0 <= x < frame.shape[1]
                        and
                        0 <= y < frame.shape[0]
                    ):

                        cv2.circle(
                            frame,
                            (x, y),
                            1,
                            (0, 255, 0),
                            -1
                        )


                # ====================================================
                # 15. HIỂN THỊ TRẠNG THÁI KỸ THUẬT
                # ====================================================

                cv2.putText(
                    frame,
                    f"Eye: {eye_status}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    f"Head: {head_direction}",
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    f"Yawn count: {len(yawn_events)}",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )


                cv2.putText(
                    frame,
                    f"Head down: {len(head_down_events)}",
                    (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )


                # ====================================================
                # 16. HIỂN THỊ HÀNH VI MẤT TẬP TRUNG
                # ====================================================

                warning_text = None


                # ----------------------------------------------------
                # 1. NHÌN SANG HƯỚNG KHÁC
                # ----------------------------------------------------

                if (
                    head_direction == "NHIN SANG TRAI"
                    or
                    head_direction == "NHIN SANG PHAI"
                ):

                    warning_text = (
                        "MAT TAP TRUNG: NHIN SANG HUONG KHAC"
                    )


                # ----------------------------------------------------
                # 2. NHẮM MẮT LÂU >= 5 GIÂY
                # ----------------------------------------------------

                elif (
                    eye_closed_start is not None
                    and
                    closed_duration >= EYE_CLOSED_TIME
                ):

                    warning_text = (
                        "MAT TAP TRUNG: NHAM MAT LAU"
                    )


                # ----------------------------------------------------
                # 3. CÚI ĐẦU >= 2 LẦN / 1 PHÚT
                # ----------------------------------------------------

                elif (
                    len(head_down_events)
                    >= HEAD_DOWN_REQUIRED
                ):

                    warning_text = (
                        "MAT TAP TRUNG: CUI DAU NHIEU"
                    )


                # ----------------------------------------------------
                # 4. NGÁP >= 2 LẦN / 10 PHÚT
                # ----------------------------------------------------

                elif (
                    len(yawn_events)
                    >= YAWN_REQUIRED
                ):

                    warning_text = (
                        "MAT TAP TRUNG: NGAP NHIEU"
                    )


                # ----------------------------------------------------
                # HIỂN THỊ WARNING
                # ----------------------------------------------------

                if warning_text is not None:

                    cv2.putText(
                        frame,
                        warning_text,
                        (20, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        3
                    )


        # ====================================================
        # 17. KHÔNG THẤY KHUÔN MẶT
        # ====================================================

        else:

            # ------------------------------------------------
            # Bắt đầu tính thời gian mất mặt
            # ------------------------------------------------

            if face_missing_start is None:

                face_missing_start = current_time


            missing_duration = (
                current_time -
                face_missing_start
            )


            # ------------------------------------------------
            # Không thấy mặt >= 10 giây
            # ------------------------------------------------

            if missing_duration >= FACE_ABSENCE_TIME:

                cv2.putText(
                    frame,
                    "MAT TAP TRUNG: ROI KHOI VI TRI CAMERA",
                    (20, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (0, 0, 255),
                    3
                )

            else:

                cv2.putText(
                    frame,
                    "KHONG NHIN RO MAT",
                    (20, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 0, 255),
                    3
                )


        # ====================================================
        # 18. HIỂN THỊ CAMERA
        # ====================================================

        cv2.imshow(
            "Concentration Behavior Detection",
            frame
        )


        # ====================================================
        # 19. ESC ĐỂ THOÁT
        # ====================================================

        if cv2.waitKey(5) & 0xFF == 27:

            break
# ============================================================
# 20. GIẢI PHÓNG
# ============================================================
cap.release()
cv2.destroyAllWindows()