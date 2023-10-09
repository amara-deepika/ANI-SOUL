import os
from twilio.rest import Client
def say():
    s=("caution "+animal+" is approaching\n" )*5
    import pyttsx3
    text_speech=pyttsx3.init()
    text_speech.say(s)
    text_speech.runAndWait()
def makecall(animal):
    # Find your Account SID and Auth Token at twilio.com/console
    # and set the environment variables. See http://twil.io/secure
    account_sid = 'ACXXXXXXXXXXXXXXXXX'
    auth_token = '2fXXXXXXXXXXXXXXXXXXXXXXX'
    client = Client(account_sid, auth_token)

    a=("caution \n "+animal+"\n is approaching\n" )*5
    call = client.calls.create(
                            twiml='<Response><Say voice="male" language="en"> %s </Say></Response>'%(a),
                            to='+91XXXXXXXXX',
                            from_='+1XXXXXXX8219'
                        )
    c=1
################################################################# call to the forest department
import os
from twilio.rest import Client
def makecall0(animal):
    # Find your Account SID and Auth Token at twilio.com/console
    # and set the environment variables. See http://twil.io/secure
    account_sid = 'ACXXXXXXXXXXXXX7a'
    auth_token = 'dc2XXXXXXXXXXXXXXXXXc'
    client = Client(account_sid, auth_token)

    a=("caution \n "+animal+"\n is approaching\n" )*5
    call = client.calls.create(
                            twiml='<Response><Say voice="male" language="en"> %s </Say></Response>'%(a),
                            to='+91XXXXXXXXX',
                            from_='+1XXXXXXXXX'
                        )
    c=1
      
c=0
animal=""
import numpy as np
import pandas as pd
from statistics import mode
import os
import cv2
import time
import matplotlib.pyplot as plt
from time import sleep
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
Name0=[]
for dirname, _, filenames in os.walk(r'''../admin/.kaggle/animals10/raw-img'''):
    Name0+=[dirname]

Name1=[]
for item in Name0:
    Name1+=[item[27:]]

Name2=[]
for item in Name1:
    if (item!=''):
        Name2+=[item]
                
Name3=sorted(Name2)
print(Name3)
labels=['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
Name3E=['dog','horse','elephant','butterfly','hen','cat','cow','sheep','spider','squirrel']
Name3Ex=[]
Name3Eo=[]
for item in Name3E:
    if item not in labels:
        Name3Ex+=[item]
    elif item in labels:
        Name3Eo+=[item]
print(Name3Ex)
print(Name3Eo)
weights_path = r'''c:\Users\admin\.kaggle\yolov3.weights'''
configuration_path = r'''C:\Users\admin\.kaggle\yolov3.cfg'''
probability_minimum = 0.5
threshold = 0.3
network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)
layers_names_all = network.getLayerNames()
outputlayers = [layers_names_all[i-1] for i in network.getUnconnectedOutLayers()]
def ImagePath(path):
    
    bounding_boxes = []
    confidences = []
    class_numbers = []
    image_input = path
    blob = cv2.dnn.blobFromImage(image_input, 1/255.0, (416,416), swapRB=True, crop=False)
    blob_to_show = blob[0,:,:,:].transpose(1,2,0)
    network.setInput(blob)
    output_from_network = network.forward(outputlayers)
    h,w = image_input.shape[:2]

    for result in output_from_network:
        for detection in result:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum:
                box_current = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center-(box_width/2))
                y_min = int(y_center-(box_height/2))
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)
    print(mode(class_numbers))
    global animal
    print(labels[mode(class_numbers)])
    animal=labels[mode(class_numbers)]
import cv2
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
sleep(5)
while True:
    try:
        _, frame = video.read()
        frame = cv2.flip(frame, 1)
        ImagePath(frame)
        cv2.imshow("ANIMAL DETECTION", frame)
        #a = cv2.waitKey(1)
        if animal in ['dog','horse','elephant','cow','sheep']:
            print("animal recognized")
            print("classifing the animal : more harming possibility ")
            cv2.imshow("ANIMAL DETECTION", frame)
            makecall(animal)
            makecall0(animal)
            say()
            c=1
            a = cv2.waitKey(1)
        if animal in ['butterfly','spider','squirrel']:
            cv2.imshow("ANIMAL DETECTION", frame)
            print("classifing the animal : less harming possibility ")
            print("domestic animal")
            makecall(animal)
            say()
            c=1
            a = cv2.waitKey(1)
        if a == ord('q') or c==1 :
            break
    except:
        continue
            
video.release()
cv2.destroyAllWindows()

