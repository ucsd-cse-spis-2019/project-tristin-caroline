import cv2
oriimg = cv2.imread("cafevcard.jpg")
img = cv2.resize(oriimg, (50,50))
cv2.imwrite('cafevcard5050.jpg',img)
