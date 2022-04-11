import cv2
import numpy as np

video = cv2.VideoCapture(0)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# face = cv2.CascadeClassifier("faces.xml")
# body = cv2.CascadeClassifier("body.xml")

while True:
    ret, frame = video.read()

    if ret:
        frame = cv2.resize(frame, (800, 600))
        bodyDetect,_ = hog.detectMultiScale(frame,winStride=(8,8))
        bodyDetect = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bodyDetect])

        for (xA,yA,xB,yB) in bodyDetect:
            # frame[y:y+h,x:x+h]=cv2.blur(frame[y:y+h,x:x+w],(50,50))
            cv2.rectangle(frame,(xA,yA),(xB,yB),(0,255,0),2)
        cv2.imshow('capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# After the loop release the cap object
video.release()
# Destroy all the windows
cv2.destroyAllWindows()

