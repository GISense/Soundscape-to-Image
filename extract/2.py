import cv2

vc = cv2.VideoCapture("../v_c_0_10.mp4")

if vc.isOpened():
    ret, frame = vc.read()
else:
    ret = False


# loop read video frame
i=0
while ret:
    ret, frame = vc.read()
    if i<250:
        image_path="./image/"+str(i)+".jpg"
        cv2.imwrite(image_path, frame)
    i=i+1
    cv2.waitKey(40)
