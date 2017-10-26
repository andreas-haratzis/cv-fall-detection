# library and dataset
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import cv2

all_b_results = []

def init():
    global all_b_results
    all_b_results = []


def reason(frame, originalFrame, detection_results_a, detection_results_b, roi_avg):

    # Threshold values
    vector_Magnitude_Threshold = 3
    deltaMagnitudeFallenThreshold = 0.1
    deltaRatioThreshold = 0.1
    deltaRatioFallenThreshold = 0.0015
    ratioThreshold = 0.5
    ratioFallenThreshold = 1.15
    ratioFallenThresholdB = 1.6
    aveDeltaAreaThreshold = 250
    aveDeltaAreaFallenThreshold = 13
    deltaVertMagThreshold = 3

    #some colours
    colourArr = ((255,0,0) ,(0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255),
                (255,125,0), (255,0,125), (125,255,0), (0,255,125), (125,0,255), (0,125,255), (255,255,255),)

    directions = detection_results_a.person.optical_dir_hist
    xAxisLength = len(directions)
    # angles = np.zeros(shape = (xAxisLength), dtype = float)
    lengths = np.zeros(shape = (xAxisLength), dtype = float)
    rectangles = detection_results_a.person.rect_hist
    ratios = np.zeros(shape = (xAxisLength), dtype = float)
    areas = np.zeros(shape = (xAxisLength), dtype = float)
    verticalMagnitudes = np.zeros(shape = (xAxisLength), dtype = float)

    if (xAxisLength > 0):
        (x2, y2, w2, h2) = detection_results_a.person.rect
        x = x2 * 2
        y = y2 * 2
        w = w2 * 2
        h = h2 * 2
        deltaLengths = np.zeros(shape = (xAxisLength-1), dtype = float)
        deltaAreas = np.zeros(shape = (xAxisLength-1), dtype = float)
        deltaRatios = np.zeros(shape = (xAxisLength-1), dtype = float)
        deltaVertMags = np.zeros(shape = (xAxisLength-1), dtype = float)
    if (xAxisLength > 3):
        averageAreas = np.zeros(shape = (xAxisLength-4), dtype = float)
        averageRatios = np.zeros(shape = (xAxisLength-4), dtype = float)
    if (xAxisLength > 4):
        averageDeltaAreas = np.zeros(shape = (xAxisLength-5), dtype = float)


    for i in range (0, xAxisLength):
        # riseOverRun = directions[i][1]/directions[i][0]
        # angles[i] = np.arctan(riseOverRun)
        a2b2 = ((directions[i][1] * directions[i][1]) + (directions[i][0] * directions[i][0]))

        lengths[i] = np.sqrt(a2b2)

        deltaX = rectangles[i][1]
        deltaY = rectangles[i][3]

        if (deltaY == 0):
            ratios[i] = 0
        else:
            ratios[i] = deltaX/deltaY

        if (directions[i][0] == 0):
            vertMag = 0
        else:
            verticalMagnitudes[i] = directions[i][1]/abs(directions[i][0])

        areas[i] = deltaX*deltaY

        if (i > 0):
            deltaLengths[i-1] = lengths[i] - lengths[i-1]
            deltaRatios[i-1] = ratios[i] - ratios[i-1]
            deltaAreas[i-1] = areas[i] - areas[i-1]
            deltaVertMags[i-1] = verticalMagnitudes[i] - verticalMagnitudes[i-1]
        if (i > 3):
            averageRatios[i-4] = (ratios[i] + ratios[i-1] + ratios[i-2] + ratios[i-3] + ratios[i-4])/5
            averageAreas[i-4] = (areas[i] + areas[i-1] + areas[i-2] + areas[i-3] + areas[i-4])/5
        if (i > 4):
            averageDeltaAreas[i-5] = (deltaAreas[i-1] + deltaAreas[i-2] + deltaAreas[i-3] + deltaAreas[i-4] + deltaAreas[i-5])/5

    #Draw some bounding boxes for clear testing, comment out during final product launch
    #colref = ((0,0,0),(255,255,255))
    if (xAxisLength > 0):
        cv2.rectangle(originalFrame[0], (x, y), (x + w, y + h), (255,255,255), 2)
        #for j in range (0,7):
        #    cv2.rectangle(originalFrame[0], (x - (j*2), y - (j*2)), (x + w + (j*2), y + h + (j*2)), colref[((j+1)%2)], 2)

    duringFall = False
    afterFall = False

    deltaRatioPass = False
    deltaAreasPass = False
    deltaVertMagsPass = False
    notFallenB = False

    VectorPass = False

        # quick increase in Ratio Value
    if (xAxisLength > 1):
        # print(xAxisLength)
        # print(ratios)
        #input("\n Press Enter to Continue... \n")
        print("Frame: ", xAxisLength)
        #check to see if the ratio of sides of the bounding rectangle is passing a threshold value
        if (ratios[xAxisLength-1] > ratioFallenThreshold):
            print ("FALLEN FIRST CHECK: ratios greater than 1:1")
            #cv2.rectangle(originalFrame[0], (x, y), (x + w, y + h), colourArr[0], 2)
            if (deltaRatios[xAxisLength-2] < deltaRatioFallenThreshold):
                print ("AFTER FALL: Low Delta Ratio")
                #cv2.rectangle(originalFrame[0], (x - 2, y - 2), (x + w + 2, y + h + 2), colourArr[1], 2)
                deltaRatioPass = True
            if (deltaVertMags[xAxisLength-2] < deltaMagnitudeFallenThreshold):
                print ("AFTER FALL: Low Delta Magnitude")
                #cv2.rectangle(originalFrame[0], (x - 4, y - 4), (x + w + 4, y + h + 4), colourArr[2], 2)
                deltaVertMagsPass = True
            if (xAxisLength > 5):
                if (averageDeltaAreas[xAxisLength-6] < aveDeltaAreaFallenThreshold):
                    print ("AFTER FALL: Low Average Delta Areas")
                    #cv2.rectangle(originalFrame[0], (x - 6, y - 6), (x + w + 6, y + h + 6), colourArr[3], 2)
                    deltaAreasPass = True

        if (ratios[xAxisLength - 1] < ratioFallenThresholdB and ratios[xAxisLength - 1] > ratioFallenThreshold):
            # Detection B
            person_avg = detection_results_b[1]
            if person_avg is None:
                person_avg = 0
            diff = abs(person_avg - roi_avg)
            notFallenB = (diff > 10)
            if (notFallenB):
                print("AFTER FALL: NOT fallen B")

        #check to see if the length of the Vector is passing a threshold value
        if (lengths[xAxisLength-1] > vector_Magnitude_Threshold):
            # If the vector is above a certain magnitude
            print ("Vector magnitude pass")
            if (deltaVertMags[xAxisLength-2] > deltaVertMagThreshold):
                # And the Vector is pointing downwards
                print ("FALLING: Vector direction pass")
                #cv2.rectangle(originalFrame[0], (x - 8, y - 8), (x + w + 8, y + h + 8), colourArr[4], 2)

        if (deltaRatios[xAxisLength-2] > deltaRatioThreshold):
            print ("FALLING: Delta Ratio pass")
            #cv2.rectangle(originalFrame[0], (x - 10, y - 10), (x + w + 10, y + h + 10), colourArr[5], 2)

        if (xAxisLength > 5):
            if (averageDeltaAreas[xAxisLength-6] > aveDeltaAreaThreshold):
                print ("FALLING: Average Delta Areas")
                #cv2.rectangle(originalFrame[0], (x - 12, y - 12), (x + w + 12, y + h + 12), colourArr[6], 2)

    # Make the call wether they are falling or have fallen

    # for Fallen, use a best-two-out-of-three case looking for certain values
    fallenA = ((deltaRatioPass or deltaVertMagsPass) and deltaAreasPass) or (deltaRatioPass and deltaVertMagsPass)
    if (xAxisLength > 0):
        if (fallenA):
            cv2.rectangle(originalFrame[0], (x, y), (x + w, y + h), colourArr[0], 2)
    fallen = (not notFallenB) and fallenA
    if (fallen):
        cv2.rectangle(originalFrame[0], (x, y), (x + w, y + h), colourArr[2], 8)


    # Create data
    yAxis = range(1,xAxisLength+1)
    deltaYAxis = range(1,xAxisLength)
    aveYAxis = range(1,xAxisLength-3)
    # angleData = pd.DataFrame({'frames': yAxis, 'angles': angles })
    lengthData = pd.DataFrame({'frames': yAxis, 'vector magnitude': lengths })
    ratioData = pd.DataFrame({'frames': yAxis, 'ratios of sides': ratios })
    areaData = pd.DataFrame({'frames': yAxis, 'areas': areas })

    if (xAxisLength > 0):
        lengthChangeData = pd.DataFrame({'frames': deltaYAxis, 'delta magnitude': deltaLengths })
        ratioChangeData = pd.DataFrame({'frames': deltaYAxis, 'delta ratios': deltaRatios })
        areaChangeData = pd.DataFrame({'frames': deltaYAxis, 'delta areas': deltaAreas })
    else:
        lengthChangeData = None
        ratioChangeData = None
        areaChangeData = None

    if (xAxisLength > 4):
        areaAverageData = pd.DataFrame({'frames': aveYAxis, 'average areas': averageAreas })
        ratioAverageData = pd.DataFrame({'frames': aveYAxis, 'average ratios': averageRatios })
    else:
        areaAverageData = None
        ratioAverageData = None
    # if (xAxisLength > 1):
    #     print (xAxisLength)
    #     input("Press Enter to continue...")
        # print (angles)
        # print (angleData.values)
        # print (angleData.as_matrix())


    #Detection B!
    all_b_results.append(detection_results_b)



    return originalFrame[0]#, lengthData, ratioData, areaData, lengthChangeData, areaChangeData, ratioChangeData, areaAverageData, ratioAverageData
    #return originalFrame[0], angleData, lengthData, ratioData, areaData


def end(avg):
    raw_dist = [0 if x[2] is None else abs(avg - x[1]) for x in all_b_results]
    smoothed_dist = []
    for i in range(0, len(raw_dist)):
        smoothed_dist.append(np.average(raw_dist[i:min(len(raw_dist), i + 10)]))
    angleGraph = plt.plot(smoothed_dist, marker='o', color='mediumvioletred')
    #plt.show()


#def end(lengthData, ratioData, areaData, lengthChangeData, areaChangeData, ratioChangeData, areaAverageData, ratioAverageData, startfall, endfall, videoName):
    #pass
    # angleGraph = plt.plot( 'frames', 'angles', data=angleData, marker='o', color='mediumvioletred')
    # angleGraph.append(plt.plot([startfall, startfall], [-1.5, 1.5], color='k', linestyle='-', linewidth=2))
    # angleGraph.append(plt.plot([endfall, endfall], [-1.5, 1.5], color='k', linestyle='-', linewidth=2))
    # plt.show()

    # print("Vector Magnitude Graph for video: " + videoName)
    # vectorGraph = plt.plot( 'frames', 'vector magnitude', data=lengthData, marker='o', color='red')
    # vectorGraph.append(plt.plot([startfall, startfall], [5, 0], color='k', linestyle='-', linewidth=2))
    # vectorGraph.append(plt.plot([endfall, endfall], [5, 0], color='k', linestyle='-', linewidth=2))
    # # vectorGraph.set_title("Vector Magnitudes")
    # # vectorGraph.set_xlabel("Frame")
    # # vectorGraph.set_ylabel("Magnitude")
    # plt.show()
    #
    # print("Change in magnitude Graph for video: " + videoName)
    # plt.plot( 'frames', 'delta magnitude', data=lengthChangeData, marker='o', color='mediumvioletred')
    # plt.plot([startfall, startfall], [2, 0], color='k', linestyle='-', linewidth=2)
    # plt.plot([endfall, endfall], [2, 0], color='k', linestyle='-', linewidth=2)
    # plt.show()
    #
    # print("Ratio of sides Graph for video: " + videoName)
    # ratioGraph = plt.plot( 'frames', 'ratios of sides', data=ratioData, marker='o', color='cyan')
    # ratioGraph.append(plt.plot([startfall, startfall], [2, 0], color='k', linestyle='-', linewidth=2))
    # ratioGraph.append(plt.plot([endfall, endfall], [2, 0], color='k', linestyle='-', linewidth=2))
    # # ratioGraph.set_title("Ratio of Width to Height")
    # # ratioGraph.set_xlabel("Frame")
    # # ratioGraph.set_ylabel("Ratio W/H")
    # plt.show()
    #
    # print("Average Ratio Graph for video: " + videoName)
    # plt.plot( 'frames', 'average ratios', data=ratioAverageData, marker='o', color='navy')
    # plt.plot([startfall, startfall], [1, 0], color='k', linestyle='-', linewidth=2)
    # plt.plot([endfall, endfall], [1, 0], color='k', linestyle='-', linewidth=2)
    # plt.show()
    #
    # print("Change in Ratio Graph for video: " + videoName)
    # plt.plot( 'frames', 'delta ratios', data=ratioChangeData, marker='o', color='blue')
    # plt.plot([startfall, startfall], [1, 0], color='k', linestyle='-', linewidth=2)
    # plt.plot([endfall, endfall], [1, 0], color='k', linestyle='-', linewidth=2)
    # plt.show()
    #
    # print("Area Graph for video: " + videoName)
    # areaGraph = plt.plot( 'frames', 'areas', data=areaData, marker='o', color='g')
    # areaGraph.append(plt.plot([startfall, startfall], [10000, 0], color='k', linestyle='-', linewidth=2))
    # areaGraph.append(plt.plot([endfall, endfall], [10000, 0], color='k', linestyle='-', linewidth=2))
    # # areaGraph.set_title("Area of bouncing rectangle")
    # # areaGraph.set_xlabel("Frame")
    # # areaGraph.set_ylabel("Area")
    # plt.show()
    #
    # print("Average Area Graph for video: " + videoName)
    # areaGraph = plt.plot( 'frames', 'average areas', data=areaAverageData, marker='o', color='chartreuse')
    # areaGraph.append(plt.plot([startfall, startfall], [10000, 0], color='k', linestyle='-', linewidth=2))
    # areaGraph.append(plt.plot([endfall, endfall], [10000, 0], color='k', linestyle='-', linewidth=2))
    # # areaGraph.set_title("Area of bouncing rectangle")
    # # areaGraph.set_xlabel("Frame")
    # # areaGraph.set_ylabel("Area")
    # plt.show()
    #
    # print("Chage of Area Graph for video: " + videoName)
    # plt.plot( 'frames', 'delta areas', data=areaChangeData, marker='o', color='lime')
    # # plt.plot([startfall, startfall], [np.amin(areaChangeData.values, axis = 1), np.amax(areaChangeData.values, axis = 1)], color='k', linestyle='-', linewidth=2)
    # # plt.plot([endfall, endfall], [np.amin(areaChangeData.values, axis = 1), np.amax(areaChangeData.values, axis = 1)], color='k', linestyle='-', linewidth=2)
    # plt.show()

    # plt.plot( 'frames', 'angles', data=angleData, marker='o', color='mediumvioletred')
    # plt.plot([startfall, startfall], [np.amin(angleData.values, axis = 1), np.amax(angleData.values, axis = 1)], color='k', linestyle='-', linewidth=2)
    # plt.plot([endfall, endfall], [np.amin(angleData.values, axis = 1), np.amax(angleData.values, axis = 1)], color='k', linestyle='-', linewidth=2)
    # plt.show()
    # plt.plot( 'frames', 'vector magnitude', data=lengthData, marker='o', color='red')
    # plt.plot([startfall, startfall], [np.amin(lengthData.values, axis = 1), np.amax(lengthData.values, axis = 1)], color='k', linestyle='-', linewidth=2)
    # plt.plot([endfall, endfall], [np.amin(lengthData.values, axis = 1), np.amax(lengthData.values, axis = 1)], color='k', linestyle='-', linewidth=2)
    # plt.show()
    # plt.plot( 'frames', 'ratios of sides', data=ratioData, marker='o', color='cyan')
    # plt.plot([startfall, startfall], [np.amin(ratioData.values, axis = 1), np.amax(ratioData.values, axis = 1)], color='k', linestyle='-', linewidth=2)
    # plt.plot([endfall, endfall], [np.amin(ratioData.values, axis = 1), np.amax(ratioData.values, axis = 1)], color='k', linestyle='-', linewidth=2)
    # plt.show()
    # plt.plot( 'frames', 'areas', data=areaData, marker='o', color='orange')
    # plt.plot([startfall, startfall], [np.amin(areaData.values, axis = 1), np.amax(areaData.values, axis = 1)], color='k', linestyle='-', linewidth=2)
    # plt.plot([endfall, endfall], [np.amin(areaData.values, axis = 1), np.amax(areaData.values, axis = 1)], color='k', linestyle='-', linewidth=2)
    # plt.show()
    #return originalFrame[0]
