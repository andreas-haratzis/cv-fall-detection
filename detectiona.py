import cv2
import numpy
from collections import namedtuple

prev_frame = None
hsv = None
mhi = None

class Person:
    __slots__ = ['rect', 'dir', 'dirHist']
    def __init__(self, rect, dir):
        self.rect = rect
        self.dir = dir
        self.dirHist = []
    def __repr__(self):
        from pprint import pformat
        return pformat([self.rect, self.dir], indent=4, width=1)
person = Person(None, None)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

def init():
    pass


def parse_frame(frame):
    global prev_frame
    global hsv
    global mhi
    global person

    imsToShow = []

    # Downsample for performance
    # Reduce to luminance only for optical flow
    frame_bw = frame.deionized
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
    elif person.rect is not None:
        pedcolour = (0, 255, 255)
        _, person.rect = cv2.CamShift(frame.fg, person.rect, term_crit)

    # If found, estimate person's velocity
    if person.rect is not None:
        (x, y, w, h) = person.rect
        roi = flow[y:y+h,x:x+w]
        mean = numpy.mean(roi, axis=(0, 1))
        person.dir = tuple(mean)
        person.dirHist.append(mean)

    pedframe = frame.deionized.copy()
    if person.rect is not None:
        (x, y, w, h) = person.rect
        r_start = (int(x + w/2), int(y + h/2))
        r_end = (int(r_start[0] + person.dir[0] * 10), int(r_start[1] + person.dir[1] * 10))
        cv2.rectangle(pedframe, (x, y), (x + w, y + h), pedcolour, 2)
        cv2.arrowedLine(pedframe, r_start, r_end, (255,255,255), 2)
    imsToShow.append(pedframe)

    if imsToShow is not []:
        cv2.imshow('Detection A', cv2.hconcat(imsToShow))

    prev_frame = frame_bw

    DetectionAResults = namedtuple('DetectionAResults', 'person')
    return DetectionAResults(person)

