import cv2
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
#type(face_cascade)
while(True):
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    #print (frame.shape)
    #print (frame.shape[:2])
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,5)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
        crop_bgr_image=frame[y:y+h,x:x+w]
        crop_gray_image=frame[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(crop_gray_image, 1.1, 5)
        for (ex, ey, ew, eh) in eyes:
            curr_state = True
            cv2.rectangle(crop_bgr_image, (ex, ey),
                          (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow("Face_detection",frame)
    k=cv2.waitKey(10)
    if(k==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()