import corrections
import detectiona
import detectionb
import reasoning
import cv2
import sys
import os
import numpy as np
import time

roi_avg = None
mouse_coords = []


def main():
    if len(sys.argv) <= 1:
        print('No folder with video provided, playing ALL')
        videos_dir = os.path.dirname(os.path.realpath(__file__)) + '/videos/'
        for dir in os.listdir(videos_dir):
            if not os.path.isdir(videos_dir + dir):

                continue
            play(dir)
    else:
        for i in range(1, len(sys.argv)):
            play(sys.argv[i])


def play(folder):
    print('Loading ' + folder)
    video = load(folder)

    if video is None:
        print('Failed to load video')
        sys.exit(1)

    print('Initializing...')
    corrections.init()
    detectiona.init()
    detectionb.init()
    reasoning.init()

    print('Running...')
    for frame in video:
        cleaned_frame = corrections.clean_frame(frame)
        detection_result_a = detectiona.parse_frame(cleaned_frame)
        perpendicular = detectionb.get_3d_coords(cleaned_frame, mouse_coords)
        detection_result_b = detectionb.parse_frame(cleaned_frame, perpendicular)#, NN_CONFIGS, cfg)
        result = reasoning.reason(cleaned_frame, frame, detection_result_a, detection_result_b, roi_avg)
        cv2.imshow("RESULT", result)
        #print(roi_avg, detection_result_b[1])
        #print(detection_result_b[0])
        cv2.waitKey(1)


def load(folder_path):
    path = os.path.dirname(os.path.realpath(__file__))
    vid_dir = path + '/videos/' + folder_path

    if not os.path.isdir(vid_dir):
        print('%s is empty' % vid_dir)
        return None

    # Load streams
    video = []
    counter = 0
    while os.path.isfile('%s/col_%d.png' % (vid_dir, counter)):
        col = cv2.imread('%s/col_%d.png' % (vid_dir, counter))
        dep = cv2.imread('%s/dep_%d.png' % (vid_dir, counter))
        set_roi(dep)
        frame = (col, dep)
        video.append(frame)
        counter += 1

    print('Loaded %d frames' % counter)
    return video

def mouse_callback(event, x, y, flags, params):
    #right-click event value is 2
    if event == 4:
        global mouse_coords

        #store the coordinates of the right-click event
        mouse_coords.append([x, y, len(mouse_coords)])

        #this just verifies that the mouse data is being collected
        #you probably want to remove this later
        print (mouse_coords)

def set_roi(dep):
    global roi_avg
    if roi_avg is None:
        global mouse_coords
        print("INSIDE")
        # Resize the image to half
        cv2.setMouseCallback('Output image', mouse_callback)
        frame_depth_half = cv2.resize(dep, (0,0), fx=0.5, fy=0.5)
        # Change to gray scale
        depth_gray = cv2.cvtColor(frame_depth_half, cv2.COLOR_BGR2GRAY)
        r = cv2.selectROI(frame_depth_half)
        # Crop image
        roi = np.array(depth_gray[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])])
        roi_avg = np.mean(roi)

    winname="Choose the four ends of the plane"
    cv2.namedWindow(winname)
    cv2.setMouseCallback(winname,mouse_callback)
    while(1):
        cv2.imshow(winname,frame_depth_half)
        if len(mouse_coords) == 4:
            break
        if cv2.waitKey(20) & 0xFF ==27:
            break
    cv2.destroyAllWindows()





if __name__ == "__main__":
    main()
