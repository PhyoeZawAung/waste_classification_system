import cv2
import datetime
import numpy as np

def resize_for_display(frame, width=800, height=480):
    """
    Resize frame for display while maintaining aspect ratio
    """
    h, w = frame.shape[:2]
    aspect = w / h
    
    if aspect > width / height:
        # Width is the limiting factor
        new_width = width
        new_height = int(width / aspect)
    else:
        # Height is the limiting factor
        new_height = height
        new_width = int(height * aspect)
    
    resized = cv2.resize(frame, (new_width, new_height))
    
    # Create a black canvas of the desired size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Calculate position to paste the resized image
    y_offset = (height - new_height) // 2
    x_offset = (width - new_width) // 2
    
    # Paste the resized image onto the canvas
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return canvas

def save_image(frame):
    path = f"yolo_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(path, frame)
    return path

def save_video_writer(path, fps=30, width=800, height=500):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(path, fourcc, fps, (width, height))
