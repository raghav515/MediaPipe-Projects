import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
cTime=0
pTime=0

mpFaceDetection=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
facedetection=mpFaceDetection.FaceDetection() #by default confidence is 0.5, if false detections encountered, just put so bigger value in the parenthesis

while True:
    success,img=cap.read()
    img=cv2.flip(img,1)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=facedetection.process(imgRGB)
    #print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
            #mpDraw.draw_detection(img,detection)  Instead of using the code below, we can just use this. SHORTCUT!!
            #print(id,detection)
            #print(detection.score)
            print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            h,w,c=img.shape
            bbox = int(bboxC.xmin*w), int(bboxC.ymin *
                                          h), int(bboxC.width*w), int(bboxC.height*h)
            cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS :{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
