import cv2
image = cv2.imread('gary.jpg',-1)
cv2.imshow('gary.jpg',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
