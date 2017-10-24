import corrections
import detectiona
import detectionb
import reasoning
import cv2
import sys
import os
vidlist =("b1", "b2", "fw1", "fw2", "fr1", "fr2", "obs1", "obs2", "obs3", "obs4", "obs5", "obs6", "obs7", "obs8", "shoes1", "shoes2")
fallstart = (20, 20, 68, 71, 60, 23, 60, 60, 1, 1, 58, 0, 136, 103, 48, 10)
fallend = (60, 67, 133, 116, 120, 77, 102, 119, 28, 42, 92, 0, 184, 158, 148, 124)

def main():
    count = 0
    if len(sys.argv) <= 1:
        print('No folder with video provided, playing ALL')
        videos_dir = os.path.dirname(os.path.realpath(__file__)) + '/videos/'
        for dir in os.listdir(videos_dir):
            if not os.path.isdir(videos_dir + dir):

                continue
            play(dir, count)
            count += 1;
    else:
        for i in range(1, len(sys.argv)):
            play(sys.argv[i], count)


def play(folder, count):
    print('Loading ' + folder)
    video = load(folder)

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
        result, ldata, rdata, adata, delLdata, delRdata, delAdata, aveAdata, aveRdata = reasoning.reason(cleaned_frame, frame, detection_result_a, detection_result_b)
        cv2.imshow("Result", result)
        cv2.waitKey(1)
    reasoning.end(ldata, rdata, adata, delLdata, delRdata, delAdata, aveAdata, aveRdata, fallstart[count], fallend[count], vidlist[count])


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
