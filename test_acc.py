import cv2
import time
import csv
from datetime import datetime
from picamera2 import Picamera2
from tflite_support.task import core, processor, vision
import pyttsx3
import serial

# === Text-to-Speech Setup ===
engine = pyttsx3.init()
def say(text):
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[12].id)  # You can change index
    engine.setProperty('rate', 128)
    engine.say(text)
    engine.runAndWait()

# === Serial Comm to Arduino ===
arduino = serial.Serial('/dev/ttyACM0', 9600)

# === Model and Detection Parameters ===
model = 'efficientdet_lite0_edgetpu.tflite'
num_threads = 4
max_detected_objects = 4

# === Camera Setup ===
dispW, dispH = 640, 480
picam2 = Picamera2()
picam2.preview_configuration.main.size = (dispW, dispH)
picam2.preview_configuration.main.format = 'RGB888'
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# === Object Detector Setup ===
base_options = core.BaseOptions(file_name=model, use_coral=True, num_threads=num_threads)
detection_options = processor.DetectionOptions(max_results=max_detected_objects, score_threshold=0.3)
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

# === CSV Logger Setup ===
logfile = open("detection_log.csv", "w", newline="")
logwriter = csv.writer(logfile)
logwriter.writerow(["Timestamp", "Detected", "X1", "Y1", "X2", "Y2", "Label", "Score"])

# === Display + FPS ===
font = cv2.FONT_HERSHEY_SIMPLEX
pos = (20, 60)
myColor = (255, 0, 0)
fps = 0
voice_counter = 0
tStart = time.time()

# === Main Loop ===
while True:
    im = picam2.capture_array()
    imRGB = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    imTensor = vision.TensorImage.create_from_array(imRGB)
    detections = detector.detect(imTensor)

    for obj in detections.detections:
        category = obj.categories[0]
        class_name = category.category_name
        score = category.score

        bbox = obj.bounding_box
        x1, y1 = bbox.origin_x, bbox.origin_y
        x2, y2 = x1 + bbox.width, y1 + bbox.height

        timestamp = datetime.now().strftime("%H:%M:%S")
        logwriter.writerow([timestamp, class_name == "person", x1, y1, x2, y2, class_name, score])

        # Only act on "person" detection
        if class_name == "person":
            center_x = x1 + x2 // 2
            center_y = y1 + y2 // 2

            # Draw box
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im, class_name, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)


    # FPS Counter
    tEnd = time.time()
    loopTime = tEnd - tStart
    fps = 0.9 * fps + 0.1 * 1 / loopTime
    tStart = time.time()
    cv2.putText(im, f"{int(fps)} FPS", pos, font, 1.5, myColor, 3)

    # Show Window
    cv2.imshow('Camera', im)
    if cv2.waitKey(1) == ord('q'):
        break

# === Cleanup ===
cv2.destroyAllWindows()
logfile.close()
