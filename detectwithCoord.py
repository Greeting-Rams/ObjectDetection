#followed this tutorial
#https://www.youtube.com/watch?app=desktop&v=yE7Ve3U5Slw
import os
import cv2
import time
import serial
from picamera2 import Picamera2
from coordinates import get_person_coordinates

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

#run these in terminal
#to install numpy 1.26.4
#pip install numpy==1.26.4

#to install tflite v0.4.3
#python -m pip install --upgrade tflite-support==0.4.3

model='efficientdet_lite0_edgetpu.tflite'
num_threads=4 #number of core threads, raspberry pi 4 has 4 cores
max_detected_objects=4


dispW=640
dispH=480


picam2=Picamera2()
picam2.preview_configuration.main.size=(dispW,dispH)
picam2.preview_configuration.main.format='RGB888'
picam2.preview_configuration.align()#helps stabilize size fed/obtained by camera
picam2.configure("preview")
picam2.start()



#calculating fps and ons screen display
pos=(20,60)
font=cv2.FONT_HERSHEY_SIMPLEX
height=1.5
weight=3
myColor=(255,0,0)#blue

# setting fps=0 to reduce errors
fps=0 

#setup obeject detection

base_options=core.BaseOptions(file_name=model, use_coral=True, num_threads=num_threads)
detection_options=processor.DetectionOptions(max_results=max_detected_objects, score_threshold=0.3)
options=vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector=vision.ObjectDetector.create_from_options(options)

#start time when going into loop
tStart=time.time()
frame_counter =0;
while True:
    #read image
    
    #from webcam
    #ret, im =cam.read()
    frame_counter += 1

    # If the counter is divisible by 6, capture/save
    if frame_counter % 15 == 0:
        filename = f"captures/capture_{frame_counter}.jpg"
        cv2.imwrite(filename, im)
        print(f"Saved: {filename}")
    im=picam2.capture_array()
#     im=cv2.flip(im,-1) #flips the image
    
    #following is converting and using the tflite model.
    #opencv has format of image in BGR and tf wand RGB
    imRGB=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    imTensor=vision.TensorImage.create_from_array(imRGB)#converts image to tensor for tflite
    detections=detector.detect(imTensor)#data structure containing the detections
    for detectedObjects in detections.detections:
        class_name = detectedObjects.categories[0].category_name
        # Only process 'person' label
        if class_name == "person":
            bbox = detectedObjects.bounding_box
            x1, y1 = bbox.origin_x, bbox.origin_y
            x2, y2 = x1 + bbox.width, y1 + bbox.height

            # Draw rectangle around person
            cv2.rectangle(im, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Add label text
            cv2.putText(im, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    
    if len(detections.detections)>=1:
        for detectedObjects in range(0,len(detections.detections)):
            if detections.detections[detectedObjects].categories[0].category_name=="person":
                
                
                ytop=detections.detections[detectedObjects].bounding_box.origin_y+150
                ybottom=detections.detections[detectedObjects].bounding_box.height
                x1=detections.detections[detectedObjects].bounding_box.origin_x
                x2=detections.detections[detectedObjects].bounding_box.width
                #print(ytop,ybottom,x1,x2)
                midy=int((ytop+ybottom)/2)
                midx=int((x2+x1)/2)
                #print(midy)
                #print(midx)
                #im[y,x]
                
               
    
    #following is showing the image (camera feed) and fps
    cv2.putText(im,str(int(fps))+' FPS',pos,font,height,myColor,weight) #display fps on screen
    cv2.imshow('Camera',im)# shows the captured image (mainly for debugging)
    if cv2.waitKey(1)==ord('q'):
        break
    tEnd=time.time()
    loopTime=tEnd-tStart
    fps= 0.9*fps + 0.1*1/loopTime #Low pass filter?
    print("frames per second: ", fps)
    tStart=time.time()
    #just added 

    
cv2.destroyAllWindows()

    
