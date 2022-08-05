#necassary imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import os

def distance(w2):
    d2 = (531.42*3.5)/w2
    return int(d2)


resW = 500
resH = 268
imW,imH = resW,resH

interpreter = tflite.Interpreter(model_path = 'F:/Trivia/Prototype/tflite/model/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite')
interpreter.allocate_tensors()

#getting model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
#check the type of input tensor
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    #loading image in 'frame' variable
    frame = cv2.resize(frame,(imW,imH))
    frame_resized = cv2.resize(frame,(width,height))
    input_data = np.expand_dims(frame_resized,axis = 0)

    if floating_model:
        input_data = (np.float32(input_data)-input_mean)/input_std

    #performing detection on an image
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    #retriveing detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

    for i in range(len(scores)):
        if ((scores[i] > 0.4) and (scores[i] <= 1.0)):
            if classes[i]==43:
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                distance_est = distance(xmax-xmin)
                cv2.putText(frame,str(distance_est)+" inches",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
