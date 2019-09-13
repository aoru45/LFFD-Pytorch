'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-09-10 18:22:23
@LastEditors: Aoru Xue
@LastEditTime: 2019-09-10 18:29:11
'''
import cv2 as cv
import random

if __name__ == "__main__":
    video_path = "/home/xueaoru/视频/VID_20180815_142126.mp4"
    cap = cv.VideoCapture(video_path)
    idx = 0
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        idx +=1
        if idx % 30 ==0:
            cv.imwrite("./images/{:0>6d}.jpg".format(random.randint(0,999999)),frame)
    cv.destroyAllWindows()

