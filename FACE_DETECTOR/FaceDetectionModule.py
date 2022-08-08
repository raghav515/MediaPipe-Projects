import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.5, model=0):

        self.minDetectionCon=minDetectionCon
        self.model=model

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        # by default confidence is 0.5, if false detections encountered, just put so bigger value in the parenthesis
        self.facedetection = self.mpFaceDetection.FaceDetection()

    def findFaces(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.facedetection.process(imgRGB)
        #print(results)
        bboxs=[]
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                bbox = int(bboxC.xmin*w), int(bboxC.ymin *
                                              h), int(bboxC.width*w), int(bboxC.height*h)
                bboxs.append([id,bbox,detection.score])
                if draw:
                    img=self.dwg(img,bbox)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return img,bboxs
    
    def dwg(self,img,bbox,l=30,t=5,rt=1):
        x,y,w,h=bbox
        x1,y1=x+w,y+h
        #cv2.rectangle(img, bbox, (255, 0, 255), rt)
        #Top Left
        cv2.line(img,(x,y),(x+l,y),(255,0,255),t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        #Top Right
        cv2.line(img, (x1, y), (x1-l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        #Bottom Left
        cv2.line(img, (x, y1), (x+l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1-l), (255, 0, 255), t)
        #Bottom Right
        cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1-l), (255, 0, 255), t)
        return img

def main():
    cap = cv2.VideoCapture(0)
    cTime = 0
    pTime = 0
    detector=FaceDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img,bls=detector.findFaces(img)
        print(bls)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS :{int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__=="__main__":
    main()
