from time import time
import cv2
from hand_detector import HandDetector

cap = cv2.VideoCapture(0)

detector = HandDetector()

cTime = 0
pTime = 0 
while True:
    success, img = cap.read()
    
    hand_landmarks, img = detector.detect(img)
    
    cTime = time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,60), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    
    cv2.imshow("Img", img)
    
    cv2.waitKey(1)