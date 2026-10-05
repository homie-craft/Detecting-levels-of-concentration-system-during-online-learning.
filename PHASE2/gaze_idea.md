# XÁC ĐỊNH HƯỚNG NHÌN CỦA MẮT (EYE TRACKING) GỒM CÁC VỊ TRÍ SAU:
- CENTER (Nhìn thẳng)   
- LEFT (Nhìn sang trái)   
- RIGHT (Nhìn sang phải)   
- UP (Nhìn lên trên)   
- DOWN (Nhìn xuống dưới)   
- Và các trạng thái kết hợp (VD: LEFT UP, RIGHT DOWN)   
# CÁC BIẾN SỬ DỤNG:
- Tọa độ các điểm (x,y) được lấy từ face_landmarks và nhân với kích thước ảnh (frame_width, frame_height) để chuyển sang tọa độ pixel thực tế.   
- Khai báo các landmark của mắt trái:
+ LEFT_IRIS = [474, 475, 476, 477] (4 điểm của mống mắt/tròng đen)   
+ LEFT_EYE_CORNERS = [362, 263] (2 điểm khóe mắt)   
+ LEFT_EYE_VERTICAL = [386, 374] (Điểm trên và dưới của mắt)   
- Khai báo các landmark của mắt phải:
+ RIGHT_IRIS = [469, 470, 471, 472] (4 điểm của mống mắt/tròng đen)   
+ RIGHT_EYE_CORNERS = [33, 133] (2 điểm khóe mắt)   
+ RIGHT_EYE_VERTICAL = [159, 145] (Điểm trên và dưới của mắt)   
# CÔNG THỨC KHOẢNG CÁCH VÀ TÌM TÂM:
- Công thức tính khoảng cách:
distance = sqrt((P2.x - P1.x)**2 + (P2.y - P1.y)**2) 
- Xác định tâm của Iris (tròng đen):
iris_center_x = sum(iris_x) / len(iris_points)   
iris_center_y = sum(iris_y) / len(iris_points)   
# CÔNG THỨC TÍNH TỶ LỆ TRÒNG ĐEN:
- Tính chiều rộng và chiều cao của hốc mắt:
eye_width = distance(corner_1, corner_2)   
eye_height = distance(top, bottom)   
- Công thức tính vị trí tương đối của Iris trên mỗi mắt:
+ iris_horizontal_ratio = distance(iris_center, corner_1) / eye_width   
+ iris_vertical_ratio = distance(iris_center, top) / eye_height   
+ Lấy trung bình tỷ lệ của cả 2 mắt để làm kết quả cuối cùng:
horizontal_ratio = (left_horizontal + right_horizontal) / 2   
vertical_ratio = (left_vertical + right_vertical) / 2   
# ĐẶT NGƯỠNG: 
- Ngưỡng biên ngang (xác định quay trái/phải): 
+ HORIZONTAL_LEFT_THRESHOLD = 0.58   
+ HORIZONTAL_RIGHT_THRESHOLD = 0.42   
- Ngưỡng biên dọc (xác định nhìn lên/xuống):
+ VERTICAL_UP_THRESHOLD = 0.40   
+ VERTICAL_DOWN_THRESHOLD = 0.60   
# ĐIỀU KIỆN XÁC ĐỊNH LEFT / RIGHT / UP / DOWN:
- Đặt điều kiện lệch theo chiều ngang (Left / Right):
if horizontal_ratio < HORIZONTAL_RIGHT_THRESHOLD:
    horizontal_direction = "RIGHT"
elif horizontal_ratio > HORIZONTAL_LEFT_THRESHOLD:
    horizontal_direction = "LEFT"
- Đặt điều kiện lệch theo chiều dọc (Up / Down):
if vertical_ratio < VERTICAL_UP_THRESHOLD:
    vertical_direction = "UP"
elif vertical_ratio > VERTICAL_DOWN_THRESHOLD:
    vertical_direction = "DOWN"
- Nếu tỷ lệ không nằm ngoài các ngưỡng trên, không có biến thiên trái, phải, lên hoặc xuống thì hệ thống sẽ mặc định trả về trạng thái đang nhìn thẳng: "CENTER". (Nếu cả 2 hướng đều lệch, hệ thống sẽ nối chuỗi để trả về dạng kết hợp, ví dụ: "LEFT UP"). 