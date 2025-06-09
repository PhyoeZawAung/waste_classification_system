from ultralytics import YOLO
import os
import sys
import cv2

class YOLODetector:
    def __init__(self, model_path="best.onnx"):
        self.confidence_threshold = 0.6  # Default confidence threshold
        
        try:
            print(f"Loading model from: {model_path}")
            if not os.path.exists(model_path):
                print(f"Error: Model file not found at {model_path}")
                print("Please ensure the model file exists in the correct location.")
                sys.exit(1)
                
            self.model = YOLO(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Please ensure the model file is a valid YOLO ONNX model.")
            sys.exit(1)

    def predict(self, frame):
        if frame is None:
            print("Error: Input frame is None")
            return None, []
            
        try:
            # Ensure frame is in correct format
            if len(frame.shape) != 3:
                print("Error: Invalid frame format")
                return frame, []
                
            results = self.model(frame, conf=self.confidence_threshold)[0]
            if not results or len(results) == 0:
                return frame, []
                
            # Return annotated frame and list of detected objects
            annotated_frame = results.plot()
            detected_objects = []
            
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = r
                class_name = results.names[int(class_id)]
                
                if confidence >= self.confidence_threshold:
                    detected_objects.append({
                        'class': class_name,
                        'confidence': confidence,
                        'box': [x1, y1, x2, y2]
                    })
            
            return annotated_frame, detected_objects
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return frame, []  # Return original frame and empty list on error
