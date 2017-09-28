import corrections
import detectiona
import detectionb
import reasoning
import cv2
import sys
import os

def main():
    if len(sys.argv) <= 0:
        print('Please provide a folder video to load')
        sys.exit(1)

    print('Loading video')
    video = load(sys.argv[1])

    if video is None:
        print('Failed to load video')
        sys.exit(1)

    print('Initializing')
    corrections.init()
    detectiona.init()
    detectionb.init()
    reasoning.init()

    print('Running')
    for frame in video:
        cleaned_frame = corrections.clean_frame(frame)
        detection_result_a = detectiona.parse_frame(cleaned_frame)
        detection_result_b = detectionb.parse_frame(cleaned_frame)
        result = reasoning.reason(cleaned_frame, frame, detection_result_a, detection_result_b)
        cv2.imshow("Result", result)
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
