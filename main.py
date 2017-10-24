import corrections
import detectiona
import detectionb
import reasoning
import cv2
import sys
import os

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
        #detection_result_a = detectiona.parse_frame(cleaned_frame)
        detection_result_b = detectionb.parse_frame(cleaned_frame)#, NN_CONFIGS, cfg)
        #result = reasoning.reason(cleaned_frame, frame, detection_result_a, detection_result_b)
        cv2.imshow("RESULT", detection_result_b)
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
        frame = (col, dep)
        video.append(frame)
        counter += 1

    print('Loaded %d frames' % counter)
    return video

if __name__ == "__main__":
    main()
