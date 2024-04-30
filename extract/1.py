import cv2
import os
# 其它格式的图片也可以
img_array = []
path = "C:/Users/hasee/Downloads/Video/extract/image/"  # 图片文件路径 # 获取该目录下的所有文件名
for i in range(250):
    #挨个读取图片
    img = cv2.imread(path+str(i)+".jpg")
    #获取图片高，宽，通道数信息
    height, width, layers = img.shape
    #设置尺寸
    size = (width, height)
    #将图片添加到一个大“数组中”
    img_array.append(img)
print("this is ok")
# avi：视频类型，mp4也可以
# cv2.VideoWriter_fourcc(*'DIVX')：编码格式，不同的编码格式有不同的视频存储类型
# fps：视频帧率
# size:视频中图片大小
fps=25
videopath='C:/Users/hasee/Downloads/Video/extract/test10.avi'#图片保存地址及格式
out1 = cv2.VideoWriter(videopath,cv2.VideoWriter_fourcc(*'DIVX'),fps, size)
for i in range(len(img_array)):
    #写成视频操作
    out1.write(img_array[i])
out1.release()
print("all is ok")

