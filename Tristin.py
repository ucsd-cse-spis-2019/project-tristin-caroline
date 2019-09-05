import cv2
import numpy as np
#from google_speech import Speech
"""
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar = dict()
for i in range(1,6):
    cifar.update(unpickle("cifar-10-batches-py/data_batch_%d" % i))
#print(type(cifar[b'data']))

cam = cv2.VideoCapture(0)
cv2.namedWindow("frame")
img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow('frame', frame)
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    sobelx = cv2.Sobel(frame,cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(frame,cv2.CV_64F, 0, 1, ksize=5)
    edges = cv2.Canny(frame, 100, 200)

    cv2.imshow('edges', edges)
    #cv2.imshow('laplacian', laplacian)
    #cv2.imshow('sobelx', sobelx)
    #cv2.imshow('sobely', sobely)

    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

img_rgb = cv2.imread('opencv-template-matching-python-tutorial.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('opencv-template-for-matching.jpg',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.7
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

cv2.imshow('Detected',img_rgb)
cv2.waitKey(0)
"""
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
iphone_cascade = cv2.CascadeClassifier('iphonexrcascade15stages.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray)
    iphones = iphone_cascade.detectMultiScale(gray, 8, 8)

    for (x,y,w,h) in iphones:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img, 'iPhone', (x-w, y-h), font, 0.5, (0,255,255), 2, cv2.LINE_AA)
    #for (x,y,w,h) in faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in eyes:
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
