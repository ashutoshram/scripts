import cv2
import os
import sys

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("capture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("capture", (1080, 720))


class_ = sys.argv[1]
img_counter = 0
while True:
    ret, frame = cam.read()
    cv2.putText(frame, class_, (50,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.imshow("capture", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "{}_{}.png".format(class_, img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1


cam.release()

cv2.destroyAllWindows()
