import cv2
oriimg = cv2.imread("pencil.jfif")
img = cv2.resize(oriimg, (50,50))
cv2.imwrite('pencil5050.jpg',img)
