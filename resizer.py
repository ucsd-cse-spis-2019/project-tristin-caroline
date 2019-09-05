import cv2
oriimg = cv2.imread("USD.jpg")
img = cv2.resize(oriimg, (50,50))
cv2.imwrite('USD5050.jpg',img)
