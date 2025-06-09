# Waste Classification System

This is a waste classification system that uses a custom YOLOv8 model for waste detection. The system can process images, videos, and webcam feeds to detect and classify various types of waste items.

## Features

- Image processing with waste object detection
- Video processing with real-time detection
- Webcam support with live detection
- Download detected images and videos
- Pause/Play functionality for videos
- Multiple webcam support
- Detection history tracking
- Supports 18 waste categories:
  - Aluminium foil
  - Bottle cap
  - Bottle
  - Broken glass
  - Can
  - Carton
  - Cigarette
  - Cup
  - Lid
  - Other litter
  - Other plastic
  - Paper
  - Plastic bag - wrapper
  - Plastic container
  - Pop tab
  - Straw
  - Styrofoam piece
  - Unlabeled litter

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your trained model:
   - Put your trained waste classification model file (waste_model.pt) in the project root directory
   - The model should be trained on the 18 waste categories listed above

4. Run the application:
```bash
python app/main.py
```

## Usage

1. **Image Processing**:
   - Click "Select Image" to choose an image file
   - The system will process the image and show detected waste items
   - Click "Download Output" to save the processed image

2. **Video Processing**:
   - Click "Select Video" to choose a video file
   - Use Play/Pause button to control video playback
   - The processed video will be saved automatically
   - Detection history will be shown in the right panel

3. **Webcam**:
   - Select your webcam from the dropdown menu
   - Click "Start Webcam" to begin live detection
   - Click "Stop" to end the webcam feed
   - Real-time detection history will be shown in the right panel

## Requirements

- Python 3.8 or higher
- OpenCV
- Pillow
- Ultralytics (YOLOv8)
- NumPy
- Custom waste classification model (waste_model.pt)

## Note

Make sure to place your trained waste classification model (waste_model.pt) in the project root directory before running the application.
