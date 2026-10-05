# XÁC ĐỊNH HÀNH VI MẮT ĐÓNG VÀ MẮT MỞ:
- Đầu tiên, xác định landmark của mắt (gồm 6 điểm) và đặt ngưỡng vi phạm
- Lập hàm tính EAR (Eye Aspect Ratio) 
- Lập hàm trả về trạng thái nhắm, mở mắt
- Sau khi đã phát hiện 1 frame có hành vi nhắm mắt/mở mắt thì sẽ hiển thị thông báo lên màn hình
# CÁC BIẾN SỬ DỤNG
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# CÔNG THỨC KHOẢNG CÁCH EUCLIDEAN: 
    distance(P1​,P2​)=sqrt((P1.x​−P2.x​)**2+(P1.y​−P2.y​)**2)
# Khoảng cách theo chiều dọc
    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
# Khoảng cách theo chiều ngang
    horizontal = distance(p1, p4)
# Công thức EAR
    ear = (vertical_1 + vertical_2)/(2.0 * horizontal)
# Tính EAR cho từng mắt
    left_ear = calculate_ear(face_landmarks,LEFT_EYE)
    right_ear = calculate_ear(face_landmarks,RIGHT_EYE)
# Tính EAR trung bình
    ear = (left_ear + right_ear) / 2
# ĐẶT NGƯỠNG:
    EAR_THRESHOLD
# ĐẶT ĐIỀU KIỆN
# Nhỏ hơn ngưỡng sẽ tính là nhắm mắt 
if ear < EAR_THRESHOLD:
    status = "NHAM MAT"
# Lớn hơn ngưỡng sẽ tính là mở mắt 
else:
    status = "MO MAT"