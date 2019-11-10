import numpy as np
import cv2

boxXMin = 20
boxXMax = 250
boxYMin = 20
boxYMax = 250

def blurAndGray(frame, blur = 11):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur, blur), 0)

    return blur

def threshold(frame, min = 70):
    ret, threshold = cv2.threshold(frame, min, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return threshold

def drawBox(frame):
    rect = cv2.rectangle(frame, (boxXMin, boxYMin), (boxXMax, boxYMax), (0, 0, 255), 1)
    text = cv2.putText(rect, 'Place hand here.', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))

    return text

def findHand(tFrame, frame):
    blackpoints = np.where(tFrame == [0])

    xmax = boxXMin
    ymin = boxYMin
    xmin = boxXMax
    ymax = boxYMax

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
    print(xmin, ymin, xmax, ymax)
    rect = cv2.rectangle(tFrame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

    roi = frame[ymin:ymax, xmin:xmax]

    midpointY = (ymax + ymin) // 2
    midpointX = (xmax + xmin) // 2

    roi2ymin = (midpointY + ymin) // 2
    roi2ymax = (midpointY + ymax) // 2
    roi2xmin = (midpointX + xmin) // 2
    roi2xmax = (midpointX + xmax) // 2

    roi2 = frame[roi2ymin:roi2ymax, roi2xmin:roi2xmax]

    return roi2

def calibrate(cap):
    backSub = cv2.createBackgroundSubtractorKNN()

    while (True):
        ret, frame = cap.read()

        mask = backSub.apply(frame)
        subFrame = cv2.bitwise_and(frame, frame, mask=mask)

        # Blur and threshold
        blur = blurAndGray(subFrame)
        tFrame = threshold(blur, 70)

        r, c, _ = frame.shape

        # Create box
        frame = drawBox(frame)

        cv2.imshow('frame', frame)
        cv2.imshow('threshold', tFrame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):
            roi2 = findHand(tFrame, frame)

            cv2.imshow('roi2', roi2)

            # hsv_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            key2 = cv2.waitKey(0) & 0xFF
            if key2 == ord('c'):
                continue
            elif key2 == ord('q'):
                break


        elif key == ord('q'):
            break