from time import time
import numpy as np
import cv2
from hands_module.hand_detector import HandDetector
from hands_module.finger_counter import FingersCounter
# tip indexes for index and middle finger in the array of landmarks
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    detector = HandDetector()
    finger_counter = FingersCounter()

    cTime = 0
    pTime = 0

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        if success:
            img = detector.detect_hand(img)
            finger_counter.count_fingers(img)

            landmarks = detector.get_landmarks_positions(img)
            if landmarks:
                ifinger_x, ifinger_y = landmarks[INDEX_FINGER_TIP][1:]
                mfinger_x, mfinger_y = landmarks[MIDDLE_FINGER_TIP][1:]
            #    print(ifinger_x,ifinger_y, mfinger_x, mfinger_y) 
            # drawing mode - drawing will be done only if one finger is up - the index finger
            # moving on the image - two fingers are up
           

            cTime = time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            cv2.imshow("Img", img)

            cv2.waitKey(1)
