import cv2
import time
import PostureEstimationModule as pm
i=int(input("enter a num"))
cap = cv2.VideoCapture(0)
detector = pm.poseDetector()
cTime = 0
pTime = 0
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmlist = detector.findPosition(img)
    if len(lmlist) !=0:
        print(lmlist[i])
        cv2.circle(img, (lmlist[i][1], lmlist[i][2]),
                   15, (0, 0, 255), cv2.FILLED)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("image", img)
    cv2.waitKey(1)
