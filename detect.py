#followed this tutorial
#https://www.youtube.com/watch?app=desktop&v=yE7Ve3U5Slw

import cv2
import time
from picamera2 import Picamera2

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

#max resolution
#w=1920, h=1080

#supported low resolutions
#w=640,H=360
#w=640,H=480
#w=480,H=320
#w=320,H=240
#w=240,H=160
#w=160,H=200

#reducing the resolution reduces the field of view
#can tweak and make smaller depending of needs/testing

dispW=640
dispH=480


picam2=Picamera2()
picam2.preview_configuration.main.size=(dispW,dispH)
picam2.preview_configuration.main.format='RGB888'
picam2.preview_configuration.align()#helps stabilize size fed/obtained by camera
picam2.configure("preview")
picam2.start()

#for webcam

#v4l2-ctl --list-devices
#the webcam should be under something with WEBCAM

#webCam='/dev/video1'
#cam=cv2.VideoCapture(webCam)
#cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
#cam.set(cv2.CAP_PROP_FRAME_HEIGTH, dispH)
#cam.set(cv2.CAP_PROP_FRAME_FPS, 30)

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
    if frame_counter % 6 == 0:
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
    #trying to see how to print, or get, data from detected images
  
    
    #time.sleep(10) #for seeing the detections data structure printed
    #print(detections)
        

#DetectionResult(detections=[Detection(bounding_box=BoundingBox(origin_x=413, origin_y=340, width=26, height=68), categories=[Category(index=43, score=0.4140625, display_name='', category_name='bottle')]), Detection(bounding_box=BoundingBox(origin_x=370, origin_y=396, width=123, height=81), categories=[Category(index=61, score=0.4140625, display_name='', category_name='chair')]), Detection(bounding_box=BoundingBox(origin_x=313, origin_y=352, width=28, height=54), categories=[Category(index=43, score=0.39453125, display_name='', category_name='bottle')]), Detection(bounding_box=BoundingBox(origin_x=478, origin_y=341, width=17, height=52), categories=[Category(index=43, score=0.33203125, display_name='', category_name='bottle')])])
 #categories=[Category(index=43, score=0.33203125, display_name='', category_name='bottle'
   
   #print(detections.detections)
#[Detection(bounding_box=BoundingBox(origin_x=139, origin_y=357, width=21, height=45), categories=[Category(index=43, score=0.43359375, display_name='', category_name='bottle')]), Detection(bounding_box=BoundingBox(origin_x=313, origin_y=354, width=27, height=53), categories=[Category(index=43, score=0.43359375, display_name='', category_name='bottle')]), Detection(bounding_box=BoundingBox(origin_x=318, origin_y=397, width=204, height=81), categories=[Category(index=61, score=0.37109375, display_name='', category_name='chair')]), Detection(bounding_box=BoundingBox(origin_x=597, origin_y=359, width=30, height=59), categories=[Category(index=43, score=0.33203125, display_name='', category_name='bottle')]), Detection(bounding_box=BoundingBox(origin_x=327, origin_y=94, width=21, height=54), categories=[Category(index=43, score=0.3125, display_name='', category_name='bottle')])]
    #print (detections.detections)
   
            
            
            
    #Detection(bounding_box=BoundingBox(origin_x=314, origin_y=354, width=25, height=53), categories=[Category(index=43, score=0.4140625, display_name='', category_name='bottle')])
    
    #print(detections.detections[1].categories)
    #[Category(index=43, score=0.37109375, display_name='', category_name='bottle')]
    
    #print(detections.detections[1].categories[0].category_name)
    #bottle
    #print(detections.detections[1].categories[0].score)
    
    #print(detections.detections[1].bounding_box.origin_x)
    #314
    







    #debugging purpose (overlays im with detected results)
    image=utils.visualize(im, detections)#decorate the image? labels w/ bounding boxes?
    
    
 
    
    
    #for x in range(xmin, xmax, increment)
    
    #for i in range(0, len(detect))
    
    if len(detections.detections)>=1:
        for detectedObjects in range(0,len(detections.detections)):
            if detections.detections[detectedObjects].categories[0].category_name=="person":
                #print(detections.detections[detectedObjects].bounding_box)
                #time.sleep(10)
                #BoundingBox(origin_x=-1, origin_y=4, width=574, height=471)
                
                #ytop=origin y+150
                #ybottom=height
                
                #midpt is (x2+x1)/2
                
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
                
                #need 2 for loops for averaging the colors, ran out of time
                #only extracting 1 pixel value
                color=im[midy,midx]
                blue=int(color[0])
                green=int(color[1])
                red=int(color[2])
                #print("blue is ", blue)
                #print("green is ", green)
                #print("red is ", red)
                
                outfitColor=None
                
                
                if red<40:
                    if green<40 and blue<40:
                        outfitColor="black"
                    if red<20:
                        if 50<green<160 and blue>200:
                            outfitColor="blue"
                        elif  green>120 and blue<160:
                            outfitColor="green"
                #else if to red<40
                elif red>100:
                    if green>130 and blue>130:
                        outfitColor="white"
                    elif green<50 and blue<100:
                        outfitColor="red"
                else:
                    outfitColor=None
                
                print("outfit color is: ", outfitColor)
                time.sleep(5)
                
                break #for breaking out of for loop (iterating through detections)
            
    
    
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
    
    
cv2.destroyAllWindows()

    
