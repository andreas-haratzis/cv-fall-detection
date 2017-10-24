import os
import sys
import cv2
import numpy as np
from scipy.misc import imread
from sklearn.decomposition import PCA
import copy
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from mpl_toolkits.mplot3d import Axes3D 
import mpl_toolkits.mplot3d.art3d as art3d

from plyfile import PlyData, PlyElement

def init():
    pass


def parse_frame(frame):
    depth_gray = cv2.cvtColor(frame[5], cv2.COLOR_BGR2GRAY)
    pixel_3d = point_cloud(depth_gray)

    # Create a 5 by 5 kernel for opening the image
    # This will get rid of small specs of noise
    kernel = np.ones((5,5))
    opening = cv2.morphologyEx(frame[4], cv2.MORPH_OPEN, kernel)
    # Create a 25 by 25 kernel for closing the image. This will join two 
    # disjoint moving objects in the image which, from the assumption, should be 
    # part of the same image
    kernel = np.ones((40,40))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # Threshold the closed image 
    ret, thresh = cv2.threshold(closing, 127, 255, 0)
    cv2.imshow("fg", thresh)
    # create a new frame
    return_frame = copy.copy(frame[2])
    # Fidn contonuous blobs of the color white in the image
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # For all the contours
    for contour in contours:
        # Filter out the really small ones
        if cv2.contourArea(contour) < 100:
            continue
        # convert the contour to an np 2d array
        np_contour = np.array([np.array(cont).flatten() for cont in contour])
        # Run it througgh the  PCA algorithm to get the axes with the most variance
        pca = PCA(n_components=2, svd_solver='full').fit(np_contour)
        # Find the length of the vectors frim the eigen values
        dist = [int(d) for d in pca.singular_values_]
        # Find the center of the contour
        center = pca.mean_
        # Find the end point of the vertical line
        vert_end = tuple([int(val) for val in center + (0.08 * pca.components_[0][0]*dist[0],\
                0.08 * pca.components_[0][1]*dist[0])])
        vert_start = tuple([int(val) for val in center - (0.08 * pca.components_[0][0]*dist[0],\
                0.08 * pca.components_[0][1]*dist[0])])
        # Find the end point of the horizontal line
        hori_start = tuple([int(val) for val in center - (0.08 * pca.components_[1][0]*dist[1],\
                0.08 * pca.components_[1][1]*dist[1])])
        hori_end = tuple([int(val) for val in center + (0.08 * pca.components_[1][0]*dist[1],\
                0.08 * pca.components_[1][1]*dist[1])])
        
        
        # Convert the center list to a tuple
        center = tuple([int(val) for val in center])
        # Compare line lengths and take the largest one
        #try:
        # Angle initialized to be -1
        try:
            vert_start_3d = np.array(pixel_3d[vert_start[0]][vert_start[1]])
            vert_end_3d = np.array(pixel_3d[vert_end[0]][vert_end[1]])

            angle = math.acos(sum(vert_end_3d * vert_start_3d)/(math.sqrt(modulo(vert_end_3d)) *\
                    math.sqrt(modulo(vert_start_3d)))) * 180/np.pi
        except:
            continue
        """
        len_line1 = math.sqrt((vert_start[0] - vert_end[0])**2 + (vert_end[1] - vert_start[1])**2)
        len_line2 = math.sqrt((hori_start[0] - hori_end[0])**2 + (hori_end[1] - hori_start[1])**2)
        if len_line1 > len_line2:
            vert_end_3d = [vert_end[0], vert_end[1], depth_gray[vert_end[0]][vert_end[1]]]
            vert_start_3d = [vert_start[0], vert_start[1], depth_gray[vert_start[0]][vert_start[1]]]
            hori_start_3d = [hori_start[0], hori_start[1], depth_gray[hori_start[0]][hori_start[1]]]
            hori_end_3d = [hori_end[0], hori_end[1], depth_gray[hori_end[0]][hori_end[1]]]
            # do something
            #angle = math.atan2(vert_end[1] - vert_start[1], 0) * 180/np.pi
            #angle = math.atan2(0, vert_end[0] - vert_start[0]) * 180/np.pi
            angle = math.atan2(vert_end[1] - vert_start[1], vert_end[0] - vert_start[0]) * 180/np.pi
        else:
            #do something else
            #angle = math.atan2(hori_end[1] - hori_start[1], 0) * 180/np.pi
            #angle = math.atan2(0, hori_end[0] - hori_start[0]) * 180/np.pi
            angle = math.atan2(hori_end[1] - hori_start[1], hori_end[0] - hori_start[0]) * 180/np.pi
        """
        # Draw the two axes
        #cv2.line(return_frame, vert_start, vert_end, (255,0,0), 4)
        #cv2.line(return_frame, hori_start, hori_end, (255,0,0), 4)
        # Draw the contours
        #cv2.drawContours(return_frame, [contour], -1, (0,255,0), 4)

    return angle
    """
    sess, inputs, outputs = nn_configs

    res = cv2.bitwise_and(frame[2],frame[2],mask = frame[4])

    image_batch = data_to_input(res)
    cv2.imshow("residual image", res) 
    # Compute prediction with the CNN
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

    # Extract maximum scoring location from the heatmap, assume 1 person
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)

    # Visualise
    return visualize.visualize_joints(frame[2], pose)
    """

def point_cloud(depth):
    """
    Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """
    fx = 234.42865568
    fy = 236.33408401
    cx = 170.8451577
    cy = 65.02577084
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth <= 255)
    z = depth
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)
    return np.dstack((x, y, z)).astype(int)

def show_cloud(pixels):
    fig = plt.figure()
    ax=fig.gca(projection='3d')

    for pixel in pixels:
        circle = Circle((pixel[0], pixel[1]), 0.01)
        ax.add_patch(circle)
        art3d.pathpatch_2d_to_3d(circle, z=pixel[2])

    plt.show()

def modulo(pixel):
    return pixel[0] **2 + pixel[1]**2 + pixel[2]**2
