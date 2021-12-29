import cv2
import mediapipe as mp


class HandDetector:
    """Hand detector class"""

    THUMB_TIP_INDEX = 4
    THUMB_THRESHOLD_LANDMARK = 3
    INDEX_FINGER_TIP_INDEX = 8
    INDEX_FINGER_THRESHOLD_LANDMARK = 6
    MIDDLE_FINGER_TIP_INDEX = 12
    MIDDLE_FINGER_THRESHOLD_LANDMARK = 10
    RING_FINGER_TIP_INDEX = 16
    RING_FINGER_THRESHOLD_LANDMARK = 14
    PINKY_TIP_INDEX = 20
    PINKY_THRESHOLD_LANDMARK = 18
    OFFSET = 4

    def __init__(self, num_hands=1, min_detection_conf=0.5, min_track_conf=0.5) -> None:
        self.num_hands = num_hands
        self.min_detection_conf = min_detection_conf
        self.min_track_conf = min_track_conf

        self._mp_model = mp.solutions.hands
        self._mp_hands = self._mp_model.Hands(
            max_num_hands=self.num_hands,
            min_detection_confidence=self.min_detection_conf,
            min_tracking_confidence=self.min_track_conf,
        )
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

    def detect_hand(self, img, draw=True):
        """Detect hand position and its landmarks"""

        # create RGB img for needs of the processing
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # process image - find landmarks
        self.results = self._mp_hands.process(rgbImg)

        if self.results.multi_hand_landmarks:
            for hand_landmark in self.results.multi_hand_landmarks:
                # draw landmarks for one hand
                if draw:
                    self._mp_draw.draw_landmarks(
                        img,
                        hand_landmark,
                        self._mp_model.HAND_CONNECTIONS,
                    )

        return img

    def get_landmarks_positions(self, img, hand_no=0):
        """Create list of landmarks positions in world space for selected hand"""
        landmarks_positions = []
        img_height, img_width, _ = img.shape

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]

            for idx, landmark in enumerate(hand.landmark):
                pos_x, pos_y = int(landmark.x * img_width), int(landmark.y * img_height)
                landmarks_positions.append((idx, pos_x, pos_y))

        return landmarks_positions

    def get_index_finger_tip(self, img, hand_no=0):
        """Track tip position of index finger of selected hand"""
        tip_position = ()
        img_height, img_width, _ = img.shape

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_no]

            index_tip_landmark = hand.landmark[self._mp_model.HandLandmark.INDEX_FINGER_TIP]
            pos_x, pos_y = int(index_tip_landmark.x * img_width), int(index_tip_landmark.y * img_height)
            tip_position = (pos_x, pos_y)

        return tip_position

    def count_fingers(self, img, hand_no=0):
        """Count number of fingers that are up"""
        landmarks = self.get_landmarks_positions(img, hand_no)
        num_fingers_up = 0
        up_fingers = [0] * 5

        # check whether the finger is up or closed based on the tip position against other finger landmark
        # if tip of the finger is below the value of landmark that is lower on the finger, finger is considered closed
        if landmarks:

            # get tips positions of all fingers - for thumb use x position, for others y position
            tips_positions = [
                landmarks[self.THUMB_TIP_INDEX][1],
                landmarks[self.INDEX_FINGER_TIP_INDEX][2],
                landmarks[self.MIDDLE_FINGER_TIP_INDEX][2],
                landmarks[self.RING_FINGER_TIP_INDEX][2],
                landmarks[self.PINKY_TIP_INDEX][2],
            ]

            thresholds_positions = [
                landmarks[self.THUMB_THRESHOLD_LANDMARK][1],
                landmarks[self.INDEX_FINGER_THRESHOLD_LANDMARK][2],
                landmarks[self.MIDDLE_FINGER_THRESHOLD_LANDMARK][2],
                landmarks[self.RING_FINGER_THRESHOLD_LANDMARK][2],
                landmarks[self.PINKY_THRESHOLD_LANDMARK][2],
            ]

            # check if the finger is up
            for finger in range(5):
                up_fingers[finger] = tips_positions[finger] < thresholds_positions[finger]

            num_fingers_up = sum(up_fingers)

        return num_fingers_up, up_fingers
