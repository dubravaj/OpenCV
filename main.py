from time import time
import cv2
from hands_module.hand_detector import HandDetector


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    cTime = 0
    pTime = 0

    while True:
        success, img = cap.read()
        if success:
            img = detector.detect_hand(img)
            tip_position = detector.get_index_finger_tip(img)
            print(tip_position)

            cTime = time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            cv2.imshow("Img", img)

            cv2.waitKey(1)
