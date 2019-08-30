import cv2
import numpy

cam = cv2.VideoCapture(0)


while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
