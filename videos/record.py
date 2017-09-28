#import the necessary modules
import freenect
import cv2
import numpy as np
import os
import time

#function to get RGB image from kinect
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array
 
#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return array
 
if __name__ == "__main__":

    path = os.path.dirname(os.path.realpath(__file__))
    viddir = path + '/' + str(int(time.time())) + '/'
    os.mkdir(viddir)

    count = 0
    while 1:

        #get a frame from RGB camera
        frame = get_video()
        #get a frame from depth sensor
        depth = get_depth()

        cv2.imwrite(viddir + 'col_' + str(count) + '.png', frame)
        cv2.imwrite(viddir + 'dep_' + str(count) + '.png', depth)
        
        #display RGB image
        cv2.imshow('RGB image',frame)
        #display depth image
        cv2.imshow('Depth image',depth)

        # quit program when 'esc' key is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
	
        count += 1

    cv2.destroyAllWindows()
