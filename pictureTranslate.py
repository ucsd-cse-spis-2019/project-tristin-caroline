import cv2
import numpy as np
from googletrans import Translator
import time
from PIL import ImageFont, ImageDraw, Image
translator = Translator()
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
iphone_cascade = cv2.CascadeClassifier('iphonexrcascade15stages.xml')

cap = cv2.VideoCapture(0)

def labelPic(x,y,w,h,color,objectName,img):
    cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "Original Word: " + objectName, (0, 30), font, 1, (0,255,255), 3, cv2.LINE_AA)
    dest, src = "ja", "en"
    transWord = translator.translate(objectName, dest, src)
    fontpath = "./simsun.ttc"     
    font = ImageFont.truetype(fontpath, 32)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((0, 420),"Translated Word: " + transWord.text, font = font, fill = (0, 255, 0))
    img = np.array(img_pil)
    cv2.imshow("res", img)
    cv2.waitKey(0)
    cv2.destroyWindow("res")
    #cv2.putText(img, transWord.text, (120, 300), font, 2, (0,255,0), 5, cv2.LINE_AA)    

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray)
    iphones = iphone_cascade.detectMultiScale(gray, 8, 8)

    for (x,y,w,h) in iphones:
        labelPic(x,y,w,h,(255,255,0),"Smartphone",img)
        #while True:
            #key = cv2.waitKey(1) or 0xff
            #if key == ord('c'):
                #break
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
