import cv2
import mediapipe as mp
import time
import handtrackingmodule as htm
i=int(input("Enter a num"))
cap = cv2.VideoCapture(0)
detector = htm.handDetector()
pTime = 0
cTime = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img)
    if len(lmlist) != 0:
            print(lmlist[i])

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("image", img)
    cv2.waitKey(1)
