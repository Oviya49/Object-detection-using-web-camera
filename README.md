# Object-detection-using-web-camera
## Aim:
To implement real-time object detection using a laptop web camera integrated with a pre-trained YOLOv4-tiny convolutional neural network through OpenCV's Deep Neural Network (DNN) module, and to visualize tracking bounding boxes with class labels and confidence scores
## Program:
```
import cv2
import numpy as np
import time

# 1. Load COCO Class Labels
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 2. Load Optimized YOLOv4-tiny Network Configuration and Weights
net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# 3. Initialize Live Webcam Feed
cap = cv2.VideoCapture(0)

print("Starting real-time object detection stream. Press Stop/Interrupt to exit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab hardware frame.")
            break
            
        # 4. Perform Forward-Pass Object Detection
        classes_idx, scores, boxes = model.detect(frame, confThreshold=0.4, nmsThreshold=0.4)
        
        # 5. Overlay Bounding Boxes and Class Labels
        for (classid, score, box) in zip(classes_idx, scores, boxes):
            label = f"{classes[classid]}: {score:.2f}"
            cv2.rectangle(frame, box, (0, 255, 0), 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Display frame output
        cv2.imshow("Real-Time YOLOv4-tiny Detection", frame)
        
        # Break loop gracefully if 'q' is pressed in window terminal
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    print("Stream stopped manually by user.")

finally:
    # 6. Safe Hardware Resource Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam successfully released.")
```
## Additional Requirements & Implementation Mechanics:
```
A. Network Architecture Selection (YOLOv4 vs YOLOv4-tiny)Standard YOLOv4 contains tens of millions of structural parameters, making it computationally heavy and causing immense lag (low frames-per-second) when running inference exclusively on a standard laptop CPU. Switching to YOLOv4-tiny drops complex deep layers in favor of a condensed architecture, drastically cutting down mathematical calculations and allowing real-time, low-latency video processing.

B. Image Preprocessing (Blob Mapping)Raw camera input metrics cannot be directly processed by neural layers. The input framework rescales image pixel values from an integer range of $[0, 255]$ to a normalized floating-point range of $[0, 1]$ via a factor of $1/255$. Concurrently, frames are uniformly resized to standard dimensions of $416 \times 416$ pixels using the cv2.dnn.blobFromImage conversion process to fit the structural dimension layer expectations of the network.

C. Overlap Filtering via Non-Maximum Suppression (NMS): During a single forward pass, the model frequently generates multiple overlapping candidate bounding regions around a single tracked target object. Non-Maximum Suppression (NMS) evaluates candidate regions using Intersection over Union (IoU) scores. It retains the bounding box possessing the absolute maximum local confidence score and suppresses redundant overlapping boxes, yielding clean, single-box object identification metrics.
```

## Result:
The real-time object detection pipeline was successfully implemented. The application successfully interfaces with the laptop camera hardware feed, resizes frame instances to mathematical blobs, and applies object inference tracking via YOLOv4-tiny weights. It accurately displays distinct object tracks (such as person and laptop) using colored bounding coordinates on live screen view screens without processing lag, shutting down cleanly upon execution termination signals.
