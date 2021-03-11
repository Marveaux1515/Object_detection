# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:18:54 2021

@author: Administrator
"""

import numpy as np
import cv2
net=cv2.dnn.readNet("YOLO obj_detection\yolov3.cfg","YOLO obj_detection\yolov3.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
with open("YOLO obj_detection\coco.names","r") as file:
    classes=[line.strip() for line in file.readlines()]
print(classes)
cap=cv2.VideoCapture(0)
while True:
    _,img=cap.read()
    height,width,_=img.shape
    
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("image",img.shape[:2])
    
    blob=cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True)
    print(blob.shape)
    """for b in blob:
        for n,img in enumerate(b):
            cv2.namedWindow(str(n),cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("image",img.shape[:2])
            cv2.imshow(str(n),img)"""
    net.setInput(blob)
    output_layers_names=net.getUnconnectedOutLayersNames()
    print(output_layers_names)
    layer_outputs=net.forward(output_layers_names)
    
    print([layer.shape for layer in layer_outputs])
    boxes=[]
    confidences=[]
    class_ids=[]
    
    
    for layer in layer_outputs:
        for detection in layer:
            scores=detection[5:]
            class_id=np.argmax(scores)
            
            confidence=scores[class_id]
            if confidence>0.3:
                x_centre=int(detection[0]*width)
                y_centre=int(detection[1]*height)
                w=int(detection[2]*width)
                h=int(detection[3]*height)
                x=int(x_centre-w/2)
                y=int(y_centre-h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    print(len(boxes))
    indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.4)
    print("indices:",indexes)
    font=cv2.FONT_HERSHEY_PLAIN
    colors=np.random.uniform(0,255,(len(boxes),3))
    if len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h=boxes[i]
            label=str(classes[class_ids[i]])
            confidence=str(round(confidences[i],2))
            color=colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,5)
            cv2.putText(img,label+" "+confidence,(x+30,y+70),font,5,(20,20,20),8)
     
    cv2.imshow("image",img)
    key=cv2.waitKey(1)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()