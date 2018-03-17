import numpy as np
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import matplotlib.pyplot as plt
import time


def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


haar_face_cascade = cv2.CascadeClassifier('/Users/Vicky/Dropbox/Projects/WebCamVideoFaceDetector/classifiers/haarcascade_frontalface_alt.xml')
img = cv2.imread('/Users/Vicky/Dropbox/Projects/WebCamVideoFaceDetector/baby.png')

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Detect faces
    faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

    



    
#print((faces))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 3)

plt.imshow(img)