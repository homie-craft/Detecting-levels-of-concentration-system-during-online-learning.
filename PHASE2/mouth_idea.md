# XÁC ĐỊNH HÀNH VI NGÁP VÀ TÍNH THỜI GIAN:
- Đầu tiên, xác định các hành vi mở miệng rộng dựa trên các landmark và đặt ngưỡng vi phạm
- Sau đó cho phân tích từng frame của camera
- Sau khi đã phát hiện 1 frame có hành vi ngáp sẽ bắt đầu tính timer (cũng đặt ngưỡng timer là bao nhiêu để tính là 1 hành vi ngáp)
- Nếu vượt ngưỡng timer => tính là 1 hành vi ngáp, thông báo lên màn hình
- Nếu ko vượt ngưỡng timer => ko tính là hành vi ngáp và reset lại timer
# CÁC BIẾN SỬ DỤNG
upper_lip = face_landmarks[13] (lấy landmark của môi trên)
lower_lip = face_landmarks[14] (lấy landmark của môi dưới)
left_mouth = face_landmarks[61] (lấy landmark của môi trái)
right_mouth = face_landmarks[291] (lấy landmark của môi phải)
# CÔNG THỨC KHOẢNG CÁCH EUCLIDEAN: 
    distance(P1​,P2​)=sqrt((P1.x​−P2.x​)**2+(P1.y​−P2.y​)**2)
# Tính chiều rộng của miệng:
vertical_distance = distance(upper_lip, lower_lip)
# Tính chiều dài của miệng:
horizontal_distance = distance(left_mouth, right_mouth)
# Tính tỷ lệ độ dài/rộng của miệng để xét:
mouth_ratio = (vertical_distance /horizontal_distance)
# ĐẶT NGƯỠNG:
MOUTH_OPEN_THRESHOLD (ngưỡng mở miệng nếu vượt qua ngưỡng mở này sẽ bị tính là 1 hành vi ngáp)
YAWN_DURATION (ngưỡng thời gian ngáp , nếu vượt qua thời gian này sẽ bị tính là 1 hành vi ngáp)
# Đặt timer
current_time = time.time()
# Nếu đây là thời điểm bắt đầu vượt ngưỡng
if mouth_ratio > MOUTH_OPEN_THRESHOLD:
# Nếu đây là lần đầu phát hiện hành vi ngáp
    if yawn_start_time is None:
        yawn_start_time = current_time
# Kiểm tra miệng đã mở đủ lâu chưa
    elif current_time - yawn_start_time >= YAWN_DURATION:
# Set cảnh báo lên màn hình 2 giây
        message_until = current_time + 2
# Reset lại timer sau khi đã thông báo xong
        yawn_start_time = None
# Khi miệng đã đóng    
    else:
        yawn_start_time = None

