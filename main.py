from time import time
import cv2
import numpy as np
from hands_module.hand_detector import HandDetector
from painter import Painter

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = HandDetector(min_detection_conf=0.75, min_track_conf=0.75)
    painter = Painter(detector)

    img_canvas = np.zeros((720, 1280, 3), np.uint8)

    state = 0

    cTime = 0
    pTime = 0

    brush_position = None
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        if success:
            img, state, brush_position = painter.paint(img, img_canvas, state, brush_position)

            cTime = time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            cv2.imshow("Img", img)

            cv2.waitKey(1)
