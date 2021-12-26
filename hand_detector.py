from time import time
import cv2
import mediapipe as mp

class HandDetector:
    """Class providing hand detector"""
    
    def __init__(self, num_hands=1, min_detection_conf=0.5, min_track_conf=0.5) -> None:
        self.num_hands = num_hands
        self.min_detection_conf = min_detection_conf
        self.min_track_conf = min_track_conf

        self._mp_model = mp.solutions.hands
        self._mp_hands = self._mp_model.Hands(max_num_hands=self.num_hands,
                            min_detection_confidence=self.min_detection_conf,
                            min_tracking_confidence=self.min_track_conf)
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        

    def detect(self, img, draw=True):
        """Detect hand position and its landmarks"""
        # create RGB img for needs of the processing
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # process image
        self.results = self._mp_hands.process(rgbImg)

        # array with landmarks coordinates for each hand
        hand_landmarks = []

        if self.results.multi_hand_landmarks:
            hand_landmarks = self.get_landmarks_positions(img, 0)
            for hand_landmark in self.results.multi_hand_landmarks:
                # draw landmarks for one hand
                if draw:
                    self._mp_draw.draw_landmarks(img, hand_landmark, self._mp_model.HAND_CONNECTIONS,
                self._mp_drawing_styles.get_default_hand_landmarks_style(),
                self._mp_drawing_styles.get_default_hand_connections_style())

        return hand_landmarks, img

    
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