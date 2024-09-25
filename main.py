import cv2 as cv
import numpy as np

cap = cv.VideoCapture('Motion detection and tracking/vtest.avi')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))


ret , frame1  = cap.read()
ret , frame2  = cap.read()
#print(frame1.shape)

while cap.isOpened():
    diff  = cv.absdiff(frame1, frame2)
    gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(gray, (5, 5), 0)

    _ , thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    dialated = cv.dilate(thresh, None, iterations=3)
    contours = cv.findContours(dialated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    

    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        if cv.contourArea(contour) < 900:
            continue
        cv.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(frame1, 'Status: Movement', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    image = cv.resize(frame1, (1280, 720))
    out.write(image)

    cv.imshow('motion detection', frame1)   
    frame1 = frame2
    ret, frame2 = cap.read()
    if cv.waitKey(60) == 27 :
       break

cv.destroyAllWindows()
cap.release()
out.release()
