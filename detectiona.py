import cv2
import numpy
from collections import namedtuple


class Person:

    __slots__ = ['rect', 'optical_dir', 'optical_dir_hist', 'rect_hist', 'rect_vel_smoothed']

    def __init__(self, rect, dir):
        self.rect = rect
        self.rect_hist = []
        self.rect_vel_smoothed = (0,0)
        self.optical_dir = dir
        self.optical_dir_hist = []

    def __repr__(self):
        from pprint import pformat
        return pformat(
            [self.rect, self.optical_dir, self.optical_dir_hist, self.rect_hist, self.rect_vel_smoothed]
            , indent=4, width=1)

prev_frame = None
hsv = None
mhi = None
person = Person(None, None)
hog = None
term_crit = None

def init():
    global prev_frame, hsv, mhi, person, hog, term_crit

    prev_frame = None
    hsv = None
    mhi = None
    person = Person(None, None)

    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    pass


def parse_frame(frame):
    global prev_frame
    global hsv
    global mhi
    global person

    imsToShow = []

    # Downsample for performance
    # Reduce to luminance only for optical flow
    frame_bw = frame.median
    frame_bw = cv2.cvtColor(frame_bw, cv2.COLOR_BGR2GRAY)

    if prev_frame is None:
        prev_frame = frame_bw
        hsv = numpy.zeros(frame_bw.shape + (3,), dtype=numpy.uint8)
        hsv[...,1] = 255
        mhi = numpy.zeros_like(frame_bw)
    else:
        # Update MHI
        temp = mhi.astype(numpy.int16)
        temp = temp - 1
        temp = temp + frame.fg
        temp[temp > 255] = 255
        temp[temp < 0] = 0
        mhi = temp.astype(numpy.uint8)
        imsToShow.append(cv2.cvtColor(mhi, cv2.COLOR_GRAY2BGR))

    # calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_frame, frame_bw, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Visualise optical flow
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/numpy.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    imsToShow.append(bgr)

    # Ped detection
    (rects, weights) = hog.detectMultiScale(frame_bw, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)

    # If we detected a person via HOG we immediately use that shape
    # If not, we use CamShift to keep tracking the persons silhouette
    if len(rects) > 0:
        pedcolour = (0, 255, 0)
        person.rect = tuple(rects[0])
        person.rect_hist.append(person.rect)
    elif person.rect is not None:
        pedcolour = (0, 255, 255)
        _, person.rect = cv2.CamShift(frame.fg, person.rect, term_crit)
        person.rect_hist.append(person.rect)

    # If found, estimate person's velocity
    if person.rect is not None:
        (x, y, w, h) = person.rect
        roi = flow[y:y + h, x:x + w]
        roi_mask = frame.fg[y:y + h, x:x + w]
        mean = cv2.mean(roi, roi_mask)
        person.optical_dir = tuple(mean)
        person.optical_dir_hist.append(mean)

    # Estimate raw rect velocity with moving average
    if len(person.rect_hist) > 4:
        avg = (0, 0)
        for i in range(0, 3):
            dir = numpy.subtract(middle(person.rect_hist[-i-1]), middle(person.rect_hist[-i]))
            person.rect_vel_smoothed = numpy.add(avg, dir)
        avg = person.rect_vel_smoothed / 4
        person.rect_vel_smoothed = avg

    # debug info
    pedframe = frame.median.copy()
    if person.rect is not None:

        # show enclosing rect
        (x, y, w, h) = person.rect
        cv2.rectangle(pedframe, (x, y), (x + w, y + h), pedcolour, 2)

        # show optical dir
        r_start = middle(person.rect)
        r_end = (int(r_start[0] + person.optical_dir[0] * 10), int(r_start[1] + person.optical_dir[1] * 10))
        cv2.arrowedLine(pedframe, r_start, r_end, (255,255,255), 2, 3)

        # show pos history
        for i in range(0, len(person.rect_hist) - 1):
            cv2.line(pedframe, middle(person.rect_hist[i]), middle(person.rect_hist[i+1]), (255, 0, 0), 2)

        # show rect dir
        if not tuple(person.rect_vel_smoothed) == (0, 0):
            a_start = middle(person.rect)
            a_end = numpy.add(a_start, person.rect_vel_smoothed)
            a_end = numpy.add(a_end, person.rect_vel_smoothed)
            cv2.arrowedLine(pedframe, tuple(a_start), (int(a_end[0]), int(a_end[1])), (128, 128, 128), 2, 3)


    imsToShow.append(pedframe)

    if imsToShow is not []:
        cv2.imshow('Detection A', cv2.hconcat(imsToShow))

    prev_frame = frame_bw

    DetectionAResults = namedtuple('DetectionAResults', 'person')
    return DetectionAResults(person)


def middle(rect):
    return int(rect[0] + rect[2] / 2), int(rect[1] + rect[3] / 2)