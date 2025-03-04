import os
import cv2

# TFLite libraries
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

# Optional utility for drawing boxes, etc.
import utils

# ----------------------------------------------
# 1. Configure the Object Detector
# ----------------------------------------------
model_path = "efficientdet_lite0_edgetpu.tflite"  # Same model you used before
base_options = core.BaseOptions(file_name=model_path, use_coral=True, num_threads=4)
detection_options = processor.DetectionOptions(max_results=5, score_threshold=0.3)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

# ----------------------------------------------
# 2. Define a function to detect objects & get coords
# ----------------------------------------------
def get_person_coordinates(image_path):
    """
    Reads an image, runs the TFLite detector,
    returns bounding-box coords for all persons found.
    """
    # Load image in BGR
    im_bgr = cv2.imread(image_path)
    if im_bgr is None:
        print(f"Could not read {image_path}, skipping.")
        return []

    # Convert to RGB (TFLite expects RGB)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

    # Create TensorImage
    tensor_image = vision.TensorImage.create_from_array(im_rgb)

    # Run inference
    detections = detector.detect(tensor_image)

    # Gather bounding boxes for "person"
    person_boxes = []
    for det in detections.detections:
        category_name = det.categories[0].category_name
        if category_name == "person":
            bbox = det.bounding_box
            # For clarity, get top-left & bottom-right corners
            x1, y1 = bbox.origin_x, bbox.origin_y
            x2, y2 = x1 + bbox.width, y1 + bbox.height
            
            # Or you might want the *center*:
            center_x = x1 + bbox.width // 2
            center_y = y1 + bbox.height // 2

            # Store whichever info is relevant; example below:
            person_boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": center_x,
                "center_y": center_y
            })

    return person_boxes

# ----------------------------------------------
# 3. Loop over all captured images
# ----------------------------------------------
if __name__ == "__main__":
    # Folder that holds your saved images
    captures_folder = "captures"

    # List all .jpg or .png in that folder
    all_files = os.listdir(captures_folder)
    image_files = [f for f in all_files if f.lower().endswith((".jpg", ".png"))]

    for img_file in sorted(image_files):
        full_path = os.path.join(captures_folder, img_file)
        print(f"\nAnalyzing: {full_path}")

        boxes = get_person_coordinates(full_path)

        if len(boxes) == 0:
            print("  -> No person detected.")
        else:
            print("  -> Person bounding boxes:")
            for idx, box in enumerate(boxes):
                print(f"     Person {idx+1}: {box}")
