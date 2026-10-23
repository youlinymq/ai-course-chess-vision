from idlelib.editor import keynames

import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

#设置中文字体
matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False

cap=cv2.VideoCapture(1)
count=0

while True:
    ret,frame=cap.read()
    if not ret:
        print("读取画面失败")
        break

    #BGR转换为HSV
    img_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # 高斯滤波去噪
    img_blur = cv2.GaussianBlur(frame, (7,7),0)
    #转灰度图
    img_gray = cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)
    #霍夫圆检测
    img_circles=cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,dp=1,minDist=60,param1=50,param2=20,
                                 minRadius=15,maxRadius=40)

    if img_circles is not None:
        img_circles=np.uint16(np.around(img_circles))
        for i in img_circles[0,:]:
            #画外圈园
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            #画圆心
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),-1)

    cv2.imshow("result",frame)
    cv2.imshow("BGR原图",frame)
    cv2.imshow("HSV图像",img_hsv)


    key = cv2.waitKey(1) & 0xFF
    if key  == ord('q'):
        break
    elif key ==ord('s'):
        cv2.imwrite(f"frame_{count}.jpg",frame)
        cv2.imwrite(f"hsv_{count}.jpg", img_hsv)
        count+=1
        print(f"图片已保存 frame_{count}.jpgq")
cv2.destroyAllWindows()
cap.release()
