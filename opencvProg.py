import numpy as np
import cv2
import configuration

cap = cv2.VideoCapture(0)

hist = configuration.calibrate(cap)
print(hist)
if hist is not None:
    while(True):
        ret, frame = cap.read()
        cv2.imshow('Frame', frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        configuration.filterImageForHand(hsv, hist)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



cap.release()
cv2.destroyAllWindows()
