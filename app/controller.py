import cv2
import threading
import time
from utils import resize_for_display, save_image, save_video_writer

class YOLOController:
    def __init__(self, detector, update_frame_callback, update_text_callback):
        self.detector = detector
        self.update_frame = update_frame_callback
        self.update_text = update_text_callback
        self.cap = None
        self.current_mode = None
        self.is_paused = False
        self.output_path = None
        self.video_writer = None
        self.processing_thread = None

    def process_image(self, image_path):
        self.stop()
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image from {image_path}")
            return
        
        self.current_mode = "image"
        annotated_frame, detected_objects = self.detector.predict(frame)
        self.update_frame(annotated_frame)
        self.update_text(detected_objects)
        
        # Save the output
        self.output_path = save_image(annotated_frame)

    def process_video(self, video_path):
        self.stop()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        self.current_mode = "video"
        self.is_paused = False
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Create output video writer
        output_filename = f"output_{int(time.time())}.mp4"
        self.video_writer = save_video_writer(output_filename, fps, width, height)
        self.output_path = output_filename
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_video_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def start_webcam(self, device_index=0):
        self.stop()
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open webcam device {device_index}")
            return
        
        self.current_mode = "webcam"
        self.is_paused = False
        
        # Get webcam properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Create output video writer
        output_filename = f"webcam_{int(time.time())}.mp4"
        self.video_writer = save_video_writer(output_filename, fps, width, height)
        self.output_path = output_filename
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_video_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _process_video_thread(self):
        while self.cap is not None and self.cap.isOpened() and self.current_mode in ["video", "webcam"]:
            if self.is_paused:
                time.sleep(0.1)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                break
                
            annotated_frame, detected_objects = self.detector.predict(frame)
            
            if self.video_writer is not None:
                self.video_writer.write(annotated_frame)
                
            self.update_frame(annotated_frame)
            self.update_text(detected_objects)
            
            # Control frame rate
            time.sleep(1/30)  # Assuming 30 FPS

    def pause_video(self):
        if self.current_mode in ["video", "webcam"]:
            self.is_paused = not self.is_paused

    def stop(self):
        self.current_mode = None
        self.is_paused = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None

    def refresh_current_frame(self):
        if self.current_mode == "image" and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                annotated_frame, detected_objects = self.detector.predict(frame)
                self.update_frame(annotated_frame)
                self.update_text(detected_objects)
