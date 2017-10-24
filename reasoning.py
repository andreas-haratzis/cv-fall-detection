# library and dataset
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def init():
    pass


def reason(frame, originalFrame, detection_results_a, detection_results_b):

    # library and dataset

    directions = detection_results_a.person.optical_dir_hist
    xAxisLength = len(directions)
    # angles = np.zeros(shape = (xAxisLength), dtype = float)
    lengths = np.zeros(shape = (xAxisLength), dtype = float)
    rectangles = detection_results_a.person.rect_hist
    ratios = np.zeros(shape = (xAxisLength), dtype = float)
    areas = np.zeros(shape = (xAxisLength), dtype = float)

    if (xAxisLength > 0):
        deltaLengths = np.zeros(shape = (xAxisLength-1), dtype = float)
        deltaAreas = np.zeros(shape = (xAxisLength-1), dtype = float)
        deltaRatios = np.zeros(shape = (xAxisLength-1), dtype = float)



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

        areas[i] = deltaX*deltaY

        if (i > 0):
            deltaLengths[i-1] = lengths[i] - lengths[i-1]
            deltaRatios[i-1] = ratios[i] - ratios[i-1]
            deltaAreas[i-1] = areas[i] - areas[i-1]

    # Create data
    yAxis = range(1,xAxisLength+1)
    deltaYAxis = range(1,xAxisLength)
    # angleData = pd.DataFrame({'frames': yAxis, 'angles': angles })
    lengthData = pd.DataFrame({'frames': yAxis, 'vector magnitude': lengths })

    ratioData=pd.DataFrame({'frames': yAxis, 'ratios of sides': ratios })

    areaData=pd.DataFrame({'frames': yAxis, 'areas': areas })

    if (xAxisLength > 0):
        lengthChangeData = pd.DataFrame({'frames': deltaYAxis, 'delta magnitude': deltaLengths })
        ratioChangeData = pd.DataFrame({'frames': deltaYAxis, 'delta ratios': deltaRatios })
        areaChangeData = pd.DataFrame({'frames': deltaYAxis, 'delta areas': deltaAreas })
    else:
        lengthChangeData = None
        ratioChangeData = None
        areaChangeData = None

    # if (xAxisLength > 1):
    #     print (xAxisLength)
    #     input("Press Enter to continue...")
        # print (angles)
        # print (angleData.values)
        # print (angleData.as_matrix())



    return originalFrame[0], lengthData, ratioData, areaData, lengthChangeData, areaChangeData, ratioChangeData
    #return originalFrame[0], angleData, lengthData, ratioData, areaData

def end(lengthData, ratioData, areaData, lengthChangeData, areaChangeData, ratioChangeData, startfall, endfall):
    # angleGraph = plt.plot( 'frames', 'angles', data=angleData, marker='o', color='mediumvioletred')
    # angleGraph.append(plt.plot([startfall, startfall], [-1.5, 1.5], color='k', linestyle='-', linewidth=2))
    # angleGraph.append(plt.plot([endfall, endfall], [-1.5, 1.5], color='k', linestyle='-', linewidth=2))
    # plt.show()

    vectorGraph = plt.plot( 'frames', 'vector magnitude', data=lengthData, marker='o', color='red')
    vectorGraph.append(plt.plot([startfall, startfall], [5, 0], color='k', linestyle='-', linewidth=2))
    vectorGraph.append(plt.plot([endfall, endfall], [5, 0], color='k', linestyle='-', linewidth=2))
    # vectorGraph.set_title("Vector Magnitudes")
    # vectorGraph.set_xlabel("Frame")
    # vectorGraph.set_ylabel("Magnitude")
    plt.show()

    plt.plot( 'frames', 'delta magnitude', data=lengthChangeData, marker='o', color='mediumvioletred')
    plt.plot([startfall, startfall], [2, 0], color='k', linestyle='-', linewidth=2)
    plt.plot([endfall, endfall], [2, 0], color='k', linestyle='-', linewidth=2)
    plt.show()

    ratioGraph = plt.plot( 'frames', 'ratios of sides', data=ratioData, marker='o', color='cyan')
    ratioGraph.append(plt.plot([startfall, startfall], [2, 0], color='k', linestyle='-', linewidth=2))
    ratioGraph.append(plt.plot([endfall, endfall], [2, 0], color='k', linestyle='-', linewidth=2))
    # ratioGraph.set_title("Ratio of Width to Height")
    # ratioGraph.set_xlabel("Frame")
    # ratioGraph.set_ylabel("Ratio W/H")
    plt.show()

    plt.plot( 'frames', 'delta ratios', data=ratioChangeData, marker='o', color='blue')
    plt.plot([startfall, startfall], [1, 0], color='k', linestyle='-', linewidth=2)
    plt.plot([endfall, endfall], [1, 0], color='k', linestyle='-', linewidth=2)
    plt.show()

    areaGraph = plt.plot( 'frames', 'areas', data=areaData, marker='o', color='g')
    areaGraph.append(plt.plot([startfall, startfall], [10000, 0], color='k', linestyle='-', linewidth=2))
    areaGraph.append(plt.plot([endfall, endfall], [10000, 0], color='k', linestyle='-', linewidth=2))
    # areaGraph.set_title("Area of bouncing rectangle")
    # areaGraph.set_xlabel("Frame")
    # areaGraph.set_ylabel("Area")
    plt.show()

    plt.plot( 'frames', 'delta areas', data=areaChangeData, marker='o', color='lime')
    # plt.plot([startfall, startfall], [np.amin(areaChangeData.values, axis = 1), np.amax(areaChangeData.values, axis = 1)], color='k', linestyle='-', linewidth=2)
    # plt.plot([endfall, endfall], [np.amin(areaChangeData.values, axis = 1), np.amax(areaChangeData.values, axis = 1)], color='k', linestyle='-', linewidth=2)
    plt.show()

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
