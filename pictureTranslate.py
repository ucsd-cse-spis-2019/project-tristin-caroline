import cv2
import numpy as np
import googletrans
from googletrans import Translator
import time
from PIL import ImageFont, ImageDraw, Image
from gtts import gTTS
from io import BytesIO
from playsound import playsound
import os

translator = Translator()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
iphone_cascade = cv2.CascadeClassifier('iphonexrcascade15stages.xml')
dollarface_cascade = cv2.CascadeClassifier('usdfacecascade15stages.xml')
fiver_cascade = cv2.CascadeClassifier('fivercascade15stages.xml')
#quarter_cascade = cv2.CascadeClassifier('quarternewcascade15stages.xml')
twenty_cascade = cv2.CascadeClassifier('twentydollarcascade15stages.xml')
cafev_cascade = cv2.CascadeClassifier('cafevcascade15stages.xml')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def labelPic(x,y,w,h,color,objectName,image):
    cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "Original Word: " + objectName, (0, 30), font, 1, (0,255,255), 3, cv2.LINE_AA)
    dest, src = "fr", "en"
    transWord = translator.translate(objectName, dest, src)
    fontpath = "NotoSansCJK-Regular.ttc"     
    font = ImageFont.truetype(fontpath, 32)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text((0, 420),"Translated Word: " + transWord.text, font = font, fill = (0, 255, 0))
    image = np.array(img_pil)
    cv2.imshow("res", image)
    tts = gTTS(objectName, lang = src)
    tts.save('original.mp3')
    tts = gTTS(transWord.text, lang = dest)
    tts.save('translated.mp3')
    while True:
        key = cv2.waitKey(1) or 0xff
        #sox_effects = ("speed", "0.5")
        if key == ord('e'):
            playsound("original.mp3")
        if key == ord('t'):
            playsound("translated.mp3")
        if key == ord('c'):
            os.remove("original.mp3")
            os.remove("translated.mp3")
            cv2.destroyWindow("res")
            #cv2.destroyAllWindows()
            break
                
    
    #cv2.putText(img, unicode(transWord.text, "utf-8"), (120, 300), font, 2, (0,255,0), 5, cv2.LINE_AA)

while True:
    ret, img = cap.read()
    cv2.imshow('img',img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray)
    iphones = iphone_cascade.detectMultiScale(gray,12,12)
    dollarfaces = dollarface_cascade.detectMultiScale(gray,5,5)
    fivers = fiver_cascade.detectMultiScale(gray,5,5)
    #quarters = quarter_cascade.detectMultiScale(gray,6,6)
    twenties = twenty_cascade.detectMultiScale(gray,5,5)
    #cafevs = cafev_cascade.detectMultiScale(gray,5,5)
    for (x,y,w,h) in iphones:
        labelPic(x,y,w,h,(255,255,0),"Smartphone",img)
    for (x,y,w,h) in dollarfaces:
        labelPic(x,y,w,h,(255,0,255),"One Dollar",img)
    for (x,y,w,h) in fivers:
        labelPic(x,y,w,h,(0,255,255),"Five dollars",img)
    #for (x,y,w,h) in quarters:
        #labelPic(x,y,w,h,(155,205,15),"Quarter dollar",img)
    for (x,y,w,h) in twenties:
        labelPic(x,y,w,h,(15,205,255),"Twenty dollars",img)
    #for (x,y,w,h) in cafevs:
        #labelPic(x,y,w,h,(15,205,255),"Dining Hall card",img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
