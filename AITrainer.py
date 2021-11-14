import numpy as np
import time
import cv2
import PoseTrackingModule as pm

count = 0.5
dir = 0
pTime = 0

detector = pm.PoseDetector()

#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(
    r"C:\Users\HP\OneDrive\Desktop\Fit\AI Trainer\curls.mp4")


while True:
    success, img = cap.read()
    # img = cv2.imread(
    # r"C:\Users\HP\OneDrive\Desktop\Fit\AI Trainer\curlstest.jpg")
    img = cv2.resize(img, (600, 600))
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    # print(lmList)
    if(len(lmList) > 0):
        angle = detector.findAngle(img, 12, 14, 16)
    per = np.interp(angle, (199, 301), (0, 100))
    bar = np.interp(angle, (199, 301), (650, 100))

    #print(angle, per)

    if per == 100:
        color = (0, 255, 0)
        if dir == 0:
            count += 0.5
            dir = 1
    if per == 0:
        color = (0, 255, 0)
        if dir == 1:
            count += 0.5
            dir = 0

    # print(count)
    cv2.rectangle(img, (0, 450), (150, 550), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(int(count)), (60, 510), cv2.FONT_HERSHEY_PLAIN, 4,
                (255, 0, 0), 5)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
