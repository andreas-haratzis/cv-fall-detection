import cv2
from collections import namedtuple

subtractor = None


def init():
    global subtractor
    subtractor = cv2.createBackgroundSubtractorKNN()


def clean_frame(frame):
    (frame_col, frame_d) = frame

    frame_half = cv2.resize(frame_col, (0,0), fx=0.5, fy=0.5)

    frame_median = cv2.medianBlur(frame_half, 5)
    cv2.imshow("Median", frame_median)
    frame_fg = subtractor.apply(frame_median)

    #frame_deionized = cv2.fastNlMeansDenoisingColored(frame_half, None, 10, 10, 5, 7)
    #cv2.imshow("Deionized", frame_deionized)
    #frame_fg = subtractor.apply(frame_deionized)

    # Remove lil islands from fg mask
    frame_fg_mask = cv2.GaussianBlur(frame_fg, (5,5), 0, 0)
    _, frame_fg_mask = cv2.threshold(frame_fg_mask, 192, 255, cv2.THRESH_BINARY)
    cv2.imshow("Foreground", frame_fg_mask)
    frame_fg = cv2.bitwise_and(frame_fg, frame_fg, mask=frame_fg_mask)

    Frame = namedtuple('Frame', 'col d half median fg')
    return Frame(frame_col, frame_d, frame_half, frame_median, frame_fg)