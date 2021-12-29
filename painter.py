from enum import IntEnum
import cv2
from hands_module.hand_detector import HandDetector


class PainterState(IntEnum):
    INITIAL_STATE = (0,)
    NO_DRAWING = (1,)
    DRAWING = (2,)
    CLEAR_IMAGE = (3,)
    CAPTURE_IMAGE = 4


class Painter:
    """Painter class providing basic painting function"""

    # tip indexes for index and middle finger in the array of landmarks
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_TIP = 12

    INDEX_FINGER_UP_IDX = 1
    MIDDLE_FINGER_UP_IDX = 2

    def __init__(self, detector: HandDetector) -> None:
        self.detector = detector

    def paint(self, img, canvas, state, brush_position=None):
        if brush_position is None:
            brush_position = (0, 0)
        draw_mode = False
        draw_color = (0, 255, 0)

        img = self.detector.detect_hand(img)
        num_fingers, up_fingers = self.detector.count_fingers(img)

        landmarks = self.detector.get_landmarks_positions(img)

        # drawing mode - drawing will be done only if one finger is up - the index finger
        # moving on the image - two fingers are up
        if landmarks:
            # state 0 = initial state, check num fingers
            if state == PainterState.INITIAL_STATE:
                if num_fingers == 2 and up_fingers[self.INDEX_FINGER_UP_IDX] and up_fingers[self.MIDDLE_FINGER_UP_IDX]:
                    state = PainterState.NO_DRAWING
                elif num_fingers == 1 and up_fingers[self.INDEX_FINGER_UP_IDX]:
                    state = PainterState.DRAWING
                elif num_fingers == 5:
                    state = PainterState.CAPTURE_IMAGE
            # index finger and middle fingers are up
            elif state == PainterState.NO_DRAWING:
                brush_position = (0, 0)
                if num_fingers == 2 and up_fingers[self.INDEX_FINGER_UP_IDX] and up_fingers[self.MIDDLE_FINGER_UP_IDX]:
                    state = PainterState.NO_DRAWING
                elif num_fingers == 1 and up_fingers[self.INDEX_FINGER_UP_IDX]:
                    state = PainterState.DRAWING
                elif num_fingers == 0:
                    state = PainterState.CLEAR_IMAGE
                elif num_fingers == 5:
                    state = PainterState.CAPTURE_IMAGE
            # only index finger is up - drawing
            elif state == PainterState.DRAWING:
                index_finger_tip_pos = landmarks[self.INDEX_FINGER_TIP][1:]
                cv2.circle(img, index_finger_tip_pos, 5, (0, 255, 0), cv2.FILLED)

                if brush_position[0] == 0 and brush_position[1] == 0:
                    brush_position = index_finger_tip_pos

                cv2.line(img, brush_position, index_finger_tip_pos, draw_color, 5)
                cv2.line(canvas, brush_position, index_finger_tip_pos, draw_color, 5)

                # set new start
                brush_position = index_finger_tip_pos
                if num_fingers == 1 and up_fingers[self.INDEX_FINGER_UP_IDX]:
                    state = PainterState.DRAWING
                elif (
                    num_fingers == 2 and up_fingers[self.INDEX_FINGER_UP_IDX] and up_fingers[self.MIDDLE_FINGER_UP_IDX]
                ):
                    state = PainterState.NO_DRAWING
                elif num_fingers == 5:
                    state = PainterState.CAPTURE_IMAGE
            # close index and middle finger - gesture to clear drawn image
            elif state == PainterState.CLEAR_IMAGE:
                canvas.fill(0)
                state = PainterState.INITIAL_STATE
            # all fingers are up
            elif state == PainterState.CAPTURE_IMAGE:
                if num_fingers == 0:
                    cv2.imwrite("drawing.png", canvas)
                    state = PainterState.CAPTURE_IMAGE
                elif (
                    num_fingers == 2 and up_fingers[self.INDEX_FINGER_UP_IDX] and up_fingers[self.MIDDLE_FINGER_UP_IDX]
                ):
                    state = PainterState.NO_DRAWING
                elif num_fingers == 1 and up_fingers[self.INDEX_FINGER_UP_IDX]:
                    state = PainterState.DRAWING

        img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
        cv2.imshow("Img", img)
        cv2.imshow("Canvas", canvas)

        return img, state, brush_position
