# XÁC ĐỊNH VỊ TRÍ ĐẦU GỒM 5 VỊ TRÍ SAU:
- Nhìn thẳng
- Quay trái
- Quay phải
- Cúi đầu
- Ngẩng lên
# CÁC BIẾN SỬ DỤNG:
- Mỗi biến sẽ có dạng (x,y,z)
Note: Chúng ta chỉ sẽ sử dụng biến x,y và biến x,y đã được chuẩn hóa chỉ trong miền giá trị [0;1] 
+ nose = face_landmarks[1] (lấy landmark của mũi)
+ left_eye = face_landmarks[33] (lấy landmark của mắt trái)
+ right_eye = face_landmarks[263] (lấy landmark của mắt phải)
+ forehead = face_landmarks[10] (lấy landmark của vùng trán/đỉnh mặt)
+ chin = face_landmarks[152] (lấy landmark của cằm)
# CÔNG THỨC KHOẢNG CÁCH: 
    distance(P1​,P2​)=sqrt((P1.x​−P2.x​)**2+(P1.y​−P2.y​)**2)
# ĐẶT NGƯỠNG:
HORIZONTAL_THRESHOLD (đây là ngưỡng biên bị lệch ngang, ta dùng để xác định khi nào đối tượng đang quay đầu sang trái hoặc bên phải)
VERTICAL_THRESHOLD (đây là ngưỡng biên bị lệch dọc, ta dùng để xác định khi nào đối tượng đang ngẩng đầu lên hoặc cúi đầu xuống)
# CÔNG THỨC XÁC ĐỊNH LEFT / RIGHT:
# Xác định trung tâm 2 mắt:
    eye_center = (left_eye.x+right_eye.x)/2 (do chỉ cần xác định trung tâm theo trục hoành nên ta chỉ sử dụng tọa độ x)
# Khoảng cách giữa 2 mắt:​​
    eye_distance = distance(left_eye,right_eye​)
# Công Thức Tính Tỷ Lệ Theo chiều ngang:
    horizontal_ratio = (nose.x - eye_center_x)/eye_distance
# Đặt điều kiện lệch trái phải
    if horizontal_ratio < -HORIZONTAL_THRESHOLD:
        return "LEFT"
    elif horizontal_ratio > HORIZONTAL_THRESHOLD:
        return "RIGHT"

    
# CÔNG THỨC XÁC ĐỊNH UP / DOWN:
# Công Thức tính trung tâm theo chiều dọc khuôn mặt
    face_center_y = (forehead.y + chin.y)/2
# Tính khoảng cách chiều dọc khuông mặt:
    face_height = abs(chin.y - forehead.y)
# Công Thức Tính Tỷ Lệ Theo chiều dọc:
vertical_ratio = (nose.y - face_center_y)/ face_height
# Set điều kiện:
    elif vertical_ratio < -VERTICAL_THRESHOLD:
        return "UP"
    elif vertical_ratio > VERTICAL_THRESHOLD:
        return "DOWN"
# Nếu ko phải các trường hợp lệch trái, phải, ngẩng đầu lên hoặc cúi đầu xuống thì ta mặc định là đang nhìn thẳng:
else:
        return "FORWARD"
