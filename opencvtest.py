import numpy as np
import cv2

cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorKNN()

while (True):
    ret, frame = cap.read()

    mask = backSub.apply(frame)

    subFrame = cv2.bitwise_and(frame, frame, mask=mask)

    # Blur and threshold
    gray = cv2.cvtColor(subFrame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)

    ret, threshold = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    r, c, _ = frame.shape

    boxXMin = boxYMin = 20
    boxXMax = boxYMax = 250
    rect = cv2.rectangle(frame, (boxXMin, boxYMin), (boxXMax, boxYMax), (0, 0, 255), 1)
    text = cv2.putText(rect, 'Place hand here.', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))

    cv2.imshow('frame', rect)
    cv2.imshow('threshold', threshold)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        print(threshold.shape)
        blackpoints = np.where(threshold == [0])
        print(len(blackpoints))

        xmax = ymax = boxXMin
        xmin = ymin = 300

        for i in range(len(blackpoints[0])):
            xcor = blackpoints[1][i]
            ycor = blackpoints[0][i]


            if xcor >= boxXMin and ycor <= boxYMax and ycor >= boxYMin and xcor <= boxXMax:
               if xcor > xmax:
                   xmax = xcor

               if ycor > ymax:
                   ymax = ycor

               if xcor < xmin:
                   xmin = xcor

               if ycor < ymin:
                   ymin = ycor

        rect = cv2.rectangle(threshold, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        cv2.imshow('threshold', threshold)

        roi = frame[ymin:ymax, xmin:xmax]
        cv2.imshow('roi', roi)
        midpointY = (ymax + ymin) // 2
        midpointX = (xmax + xmin) // 2

        roi2ymin = (midpointY + ymin) // 2
        roi2ymax = (midpointY + ymax) // 2
        roi2xmin = (midpointX + xmin) // 2
        roi2xmax = (midpointX + xmax) // 2
        roi2 = frame[roi2ymin:roi2ymax, roi2xmin:roi2xmax]
        cv2.imshow('roi2', roi2)


        hsv_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        key2 = cv2.waitKey(0) & 0xFF
        if key2 == ord('c'):
            continue
        elif key2 == ord('q'):
            break


    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
