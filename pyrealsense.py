import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2
from realsense_depth import *

point = (400, 300)

def show_distance(event, x, y, args, params):
    global point
    point = (x, y)

# Initialize Camera Intel Realsense
dc = DepthCamera()

classNames = []
classFile = '/Users/williamschill/ComputerVision/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = '/Users/williamschill/ComputerVision/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = '/Users/williamschill/ComputerVision/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean(127.5)
net.setInputSwapRB(True)

# Create mouse event
cv2.namedWindow("Color frame")
cv2.setMouseCallback("Color frame", show_distance)

while True:
    ret, depth_frame, color_frame = dc.get_frame()
    classIds, confs, bbox = net.detect(color_frame, confThreshold=0.6)

    # Show distance for a specific point
    #cv2.circle(color_frame, point, 4, (0, 0, 255))
    #distance = depth_frame[point[1], point[0]]

    #cv2.putText(color_frame, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            bboxColor = (0, 255, 0)

            if (box[0] + (box[2] // 2) < 480 and box[1] + (box[3] // 2) < 640):
                
                distance = depth_frame[box[0] + (box[2] // 2), box[1] + (box[3] // 2)]
                if (distance < 2000):
                    bboxColor = (0, 0, 255)
                cv2.putText(color_frame, "{}mm".format(distance), (box[0] + (box[2] // 2), box[1] + (box[3] // 2)), cv2.FONT_HERSHEY_PLAIN, 2, bboxColor, 2)

            cv2.rectangle(color_frame, box, color=bboxColor, thickness=2)
            cv2.putText(color_frame, classNames[classId-1].upper(), (box[0]+10, box[1]+30), 
            cv2.FONT_HERSHEY_COMPLEX, 1, bboxColor, 2)

            

    #cv2.imshow("depth frame", depth_frame)
    cv2.imshow("Color frame", color_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break