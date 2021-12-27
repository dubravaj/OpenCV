import cv2
from hands_module.hand_detector import HandDetector

class FingersCounter:
    """Fingers counter module"""

    THUMB_TIP_INDEX = 4
    THUMB_TRESHOLD_LANDMARK = 2
    INDEX_FINGER_TIP_INDEX = 8
    INDEX_FINGER_THRESHOLD_LANDMARK = 6
    MIDDLE_FINGER_TIP_INDEX = 12
    MIDDLE_FINGER_TRESHOLD_LANDMARK = 10
    RING_FINGER_TIP_INDEX = 16
    RING_FINGER_THRESHOLD_LANDMARK = 14
    PINKY_TIP_INDEX = 20
    PINKY_THRESHOLD_LANDMARK = 18

    def __init__(self, num_hands=1, min_detection_conf=0.5, min_track_conf=0.5) -> None:
        self.num_hands = num_hands
        self.min_detection_conf = min_detection_conf
        self.min_track_conf = min_track_conf

        self.hand_detector = HandDetector(self.num_hands, self.min_detection_conf, self.min_track_conf)

    def count_fingers(self, img):
        self.hand_detector.detect_hand(img)
        landmarks = self.hand_detector.get_landmarks_positions(img)

        # check whether the finger is up or closed based on the tip position against other finger landmark
        # if tip of the finger is below the value of landmark that is lower on the finger, finger is considered closed
        if landmarks:
            
            up_fingers = [0] * 5
            # get tips posi tions of all fingers
            tips_positions = [landmarks[self.THUMB_TIP_INDEX][1], landmarks[self.INDEX_FINGER_TIP_INDEX][2], landmarks[self.MIDDLE_FINGER_TIP_INDEX][2],
                landmarks[self.RING_FINGER_TIP_INDEX][2], landmarks[self.PINKY_TIP_INDEX][2]
            ]

            thresholds_positions = [landmarks[self.THUMB_TRESHOLD_LANDMARK][1], landmarks[self.INDEX_FINGER_THRESHOLD_LANDMARK][2],
            landmarks[self.MIDDLE_FINGER_TRESHOLD_LANDMARK][2], landmarks[self.RING_FINGER_THRESHOLD_LANDMARK][2], landmarks[self.PINKY_THRESHOLD_LANDMARK][2]]

            print(tips_positions)

            for finger in range(5):
                up_fingers[finger] = (tips_positions[finger] < thresholds_positions[finger])
            
            num_fingers_up = sum(up_fingers)

            print(f"Number of fingers that are up: {num_fingers_up}")
