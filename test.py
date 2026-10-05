import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# img_test = cv.imread('image test/thor.jpg')
# cv.imshow('Its just THOR', img_test)
# cv.waitKey(0)

# video_cap = cv.VideoCapture('vid test/test.mp4')
# while True:
#     isYes, frame = video_cap.read()
#     cv.imshow('This vid',frame)
#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break
# video_cap.release()
# cv.destroyAllWindows()

# img_test = cv.imread('image test/thor.jpg')
def rescale_img(frame, scale = 0.7):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimesions = (width, height)
    return cv.resize(frame, dimesions, interpolation=cv.INTER_AREA)
# video_cap = cv.VideoCapture('vid test/test.mp4')
# while True:
#     isYes, frame = video_cap.read()
#     frame_da_chinh = rescale_img(frame)
#     cv.imshow('This vid',frame)
#     cv.imshow('That vid',frame_da_chinh)
#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break
# video_cap.release()
# cv.destroyAllWindows()
# def change_Res(width, height):
#     #Live video
#     capture.set(3, width)
#     capture.set(4, height)
# blank = np.zeros((500,500,3),dtype='uint8')
# # blank[200:300,300:400]=0,0,255
# cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (255,0,0),thickness=-1)
# cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 50, (255,255,0),thickness=-1)
# cv.line(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (255,0,255),thickness=1)
# cv.putText(blank,"Hello World",(225,225),cv.FONT_HERSHEY_TRIPLEX,1.5,(0,255,0),thickness=2)
# cv.imshow("Its just color",blank)
# cv.waitKey(0)

#gray scale
img_test = cv.imread('image test/leehuy.jpg')
# gray = cv.cvtColor(img_test, cv.COLOR_BGR2GRAY)
# cv.imshow('Its just grey THOR', gray)
# cv.waitKey(0)

#blur
blur = cv.GaussianBlur(img_test, (7,7), cv.BORDER_DEFAULT)
# cv.imshow("Its just blur", blur)
# cv.waitKey(0)
#Egde Cascade
canny = cv.Canny(blur, 125,175)
# cv.imshow("Its just edging",canny)
# cv.waitKey(0)
#Dilating the img
dilated = cv.dilate(canny, (3,3), iterations=2)
# cv.imshow("Its just dilating",dilated)
# cv.waitKey(0)
#erode the img
erodd = cv.erode(dilated, (3,3), iterations=1)
# cv.imshow("Its just eroding",dilated)
# cv.waitKey(0)
#resize
# resized = cv.resize(img_test, (500,500), interpolation=cv.INTER_LINEAR)
# cv.imshow("resize time", resized)
# cv.waitKey(0)
#CROP
cropped = img_test[0:500, 200:1000]
# cv.imshow("U CROPPED", cropped)
# cv.waitKey(0)

#Translation
def translate(img, x, y):
    transMate = np.float32([[1,0,x],[0,1,y]])
    dimesions = (img.shape[1],img.shape[0])
    return cv.warpAffine(img,transMate,dimesions)
#-x: Left
#-y: Up
#x: Right
#y: Down
#Rotation1
def rotate(img, goc, rotPoint=None):
    (height, width) = img.shape[:2]
    if(rotPoint is None):
        rotPoint = (width//2, height//2)
    rotMat = cv.getRotationMatrix2D(rotPoint, goc, 1.0)
    dimesions = (width,height)
    return cv.warpAffine(img,rotMat, dimesions)
#Resizing
resized = cv.resize(img_test, (500,500), interpolation=cv.INTER_CUBIC)
#Flipping
flipping = cv.flip(img_test, -1)
translated=translate(img_test,-100,200)
rotated = rotate(img_test,-45)
#CROPPING
# cv.imshow("U CROPPED", cropped)
# cv.imshow("Trans", translated)
# cv.imshow("Rotated", rotated)
# cv.imshow("Rotated Rotated", rotate(rotated,-45))
# cv.imshow("resize again", resized)
# cv.imshow("Flipping", flipping)
gray = cv.cvtColor(img_test,cv.COLOR_BGR2GRAY)
# blur = cv.GaussianBlur(gray, (7,7), cv.BORDER_DEFAULT)
# canny = cv.Canny(img_test, 125,175)
# # cv.imshow("Its gray leehuy", gray)
# cv.imshow("Its canny Lee Huy", blur)
# ret, thresh = cv.threshold(gray, 110,240,cv.THRESH_BINARY)
# cv.imshow("Its just thresh", thresh)
blank = np.zeros(img_test.shape[:2],dtype='uint8')

# contours, hierarchies = cv.findContours(canny, cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
# cv.drawContours(blank, contours, -1, (255,0,0), 1)
# cv.imshow("Its just blank", blank)
# print(len(contours), "contour(s) found!")
# cv.waitKey(0)

# BGR->HSV
hsv = cv.cvtColor(img_test, cv.COLOR_BGR2HSV)
# cv.imshow("HSV", hsv)
# #BGR -> LAB
lab = cv.cvtColor(img_test, cv.COLOR_BGR2LAB)
# cv.imshow("LAB", lab)
# cv.waitKey(0)
# BGR -> RGB
rgb = cv.cvtColor(img_test, cv.COLOR_BGR2RGB)
# cv.imshow("RGB", rgb)
# cv.waitKey(0)
# HSV -> BGR
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
# cv.imshow("hsv->bgr", hsv_bgr)
# cv.waitKey(0)
# lab -> bgr
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
# cv.imshow("lab->bgr", lab_bgr)
# cv.waitKey(0)
# plt.imshow(hsv_bgr)
# plt.show()

# #Color channels
# b,g,r=cv.split(img_test)
# blue = cv.merge([b,blank,blank])
# green = cv.merge([blank,g,blank])
# red = cv.merge([blank,blank,r])
# cv.imshow("Blue",blue)
# cv.imshow("Green",green)
# cv.imshow("Red",red)
# print(img_test.shape)
# print(b.shape)
# print(g.shape)
# print(r.shape)

# merged = cv.merge([g,b,r])
# cv.imshow('Merged Image', merged)
# cv.waitKey(0)

#Blurring

#Averaging
average = cv.blur(img_test,(3,3))
# cv.imshow("average", average)

# Gaussian Blur
gauss = cv.GaussianBlur(img_test,(3,3),0)
# cv.imshow("GaussianBlur",gauss)

#Median Blur
median = cv.medianBlur(img_test,3)
# cv.imshow("Median Blur", median)

#Bilateral blur
bilateral=cv.bilateralFilter(img_test,10,35,25)
# cv.imshow("Bilateral Blur", bilateral)

blank = np.zeros((400,400), dtype="uint8")
hcn = cv.rectangle(blank.copy(),(30,30), (370,370), 255, -1)
hinh_tron = cv.circle(blank.copy(), (200,200),200,255,-1)
# cv.imshow("hcn",hcn)
# cv.imshow("round",hinh_tron)

#bitwise AND
bitwise_and = cv.bitwise_and(hcn,hinh_tron)
# cv.imshow("bitwiseAND", bitwise_and)
#bitwise OR
bitwise_or = cv.bitwise_or(hcn,hinh_tron)
# cv.imshow("bitwiseOR", bitwise_or)
#bitwise XOR
bitwise_xor = cv.bitwise_xor(hcn,hinh_tron)
# cv.imshow("bitwiseXOR", bitwise_xor)
#bitwise NOT
bitwise_not = cv.bitwise_not(bitwise_xor)
# cv.imshow("bitwiseNOT", bitwise_not)
#MASKING
blank = np.zeros(img_test.shape[:2], dtype="uint8")
cv.imshow("Blank", blank)
# mask = cv.circle(blank, (img_test.shape[1]//2,img_test.shape[0]//2),100,255,-1)
# mask_v2 = cv.rectangle(blank,(img_test.shape[1]//2,img_test.shape[0]//2),(img_test.shape[1]//2+50,img_test.shape[0]//2+50),100,255,-1)
circle = cv.circle(blank.copy(),(img_test.shape[1]//2 + 45, img_test.shape[0]//2),100, 255,-1)
rectangle = cv.rectangle(blank.copy(), (30,30), (370,370),255,-1)
weird_shape = cv.bitwise_and(circle,rectangle)
cv.imshow("weird",weird_shape)
# cv.imshow("Mask",mask)
# cv.imshow("Mask_v2",mask_v2)
masked = cv.bitwise_and(img_test,img_test,mask=weird_shape)
cv.imshow("Masked",masked)
cv.waitKey(0)
