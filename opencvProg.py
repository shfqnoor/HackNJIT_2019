import numpy as np
import cv2
import configuration

cap = cv2.VideoCapture(0)

configuration.calibrate(cap)

cap.release()
cv2.destroyAllWindows()
