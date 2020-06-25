
import numpy as np
import cv2
import pickle
i=0
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
labels ={}
with open("labels.pickle","rb") as f:
    old_labels = pickle.load(f)
    labels = {v:k for k,v in old_labels.items()}
    
cap=cv2.VideoCapture(0)

while(True):
    
    
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray)
     
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        id_,conf = recognizer.predict(roi_gray)
        if conf>=70:
            print(id_,labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color=(255,255,255)
            thickness = 2
            cv2.putText(frame,name,(x,y),font,1,color,thickness,cv2.LINE_AA)
            
            
                
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = "Not Recognized"
            color=(255,255,255)
            thickness = 2
            cv2.putText(frame,name,(x,y),font,1,color,thickness,cv2.LINE_AA)
            
        
        
        color = (0,0,255)
        thickness = 2
        width = x+w
        height = y+h
        cv2.rectangle(frame,(x,y),(width,height),color,thickness)      
        
         
    cv2.imshow("frame",frame)
    if cv2.waitKey(20) & 0xFF==ord("q"):
        break
    
     
cap.release()
cv2.destroyAllWindows()