import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledText
from tkinter import filedialog, messagebox
import cv2
from datetime import datetime

from detector import YOLODetector
from controller import YOLOController
from utils import resize_for_display

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Waste Classification System")
        self.root.geometry("1400x800")  # Increased height for frame details
        
        # Set theme
        self.style = ttk.Style("darkly")
        
        # Detector + Controller
        self.detector = YOLODetector()
        self.controller = YOLOController(self.detector, self.update_frame, self.update_text)

        # Main container
        self.main_container = ttk.Frame(root)
        self.main_container.pack(fill=BOTH, expand=YES, padx=10, pady=10)

        # Left control panel
        self.left_panel = ttk.LabelFrame(self.main_container, text="Controls", padding=10)
        self.left_panel.pack(side=LEFT, fill=Y, padx=(0, 10))

        # Control buttons
        self.btn_img = ttk.Button(self.left_panel, text="Select Image", 
                                 command=self.select_image, width=20)
        self.btn_img.pack(pady=5, fill=X)
        
        self.btn_video = ttk.Button(self.left_panel, text="Select Video", 
                                   command=self.select_video, width=20)
        self.btn_video.pack(pady=5, fill=X)
        
        self.btn_webcam = ttk.Button(self.left_panel, text="Start Webcam", 
                                    command=self.start_webcam, width=20)
        self.btn_webcam.pack(pady=5, fill=X)

        # Webcam selection
        self.device_var = ttk.StringVar(self.left_panel)
        self.device_dropdown = None
        self.detected_devices = self.detect_cameras()
        if self.detected_devices:
            self.device_var.set(self.detected_devices[0])
            self.device_dropdown = ttk.OptionMenu(self.left_panel, self.device_var, 
                                                *self.detected_devices)
            self.device_dropdown.pack(pady=5, fill=X)

        # Confidence threshold control
        threshold_frame = ttk.LabelFrame(self.left_panel, text="Confidence Threshold", padding=10)
        threshold_frame.pack(pady=10, fill=X)
        
        self.confidence_var = ttk.DoubleVar(value=0.6)
        self.confidence_scale = ttk.Scale(threshold_frame, from_=0.0, to=1.0, 
                                        orient=HORIZONTAL, variable=self.confidence_var,
                                        command=self.update_confidence)
        self.confidence_scale.pack(fill=X, padx=5)
        
        self.confidence_label = ttk.Label(threshold_frame, 
                                        text=f"Current: {self.confidence_var.get():.2f}")
        self.confidence_label.pack(pady=5)

        # Playback controls
        self.play_pause_btn = ttk.Button(self.left_panel, text="Pause", 
                                       command=self.toggle_play_pause, 
                                       state=DISABLED, width=20)
        self.play_pause_btn.pack(pady=5, fill=X)
        
        self.stop_btn = ttk.Button(self.left_panel, text="Stop", 
                                  command=self.stop, width=20)
        self.stop_btn.pack(pady=5, fill=X)
        
        self.download_btn = ttk.Button(self.left_panel, text="Download Output", 
                                     command=self.download_output, 
                                     state=DISABLED, width=20)
        self.download_btn.pack(pady=5, fill=X)

        # Center Display Frame
        self.display_frame = ttk.LabelFrame(self.main_container, text="Preview", padding=10)
        self.display_frame.pack(side=LEFT, fill=BOTH, expand=YES, padx=10)

        # Fixed size frame for video display
        self.video_frame = ttk.Frame(self.display_frame, width=800, height=480)
        self.video_frame.pack(pady=(0, 10))
        self.video_frame.pack_propagate(False)  # Prevent frame from shrinking

        self.frame_label = ttk.Label(self.video_frame)
        self.frame_label.pack(expand=YES)

        # Frame details section
        details_frame = ttk.LabelFrame(self.display_frame, text="Frame Details", padding=10)
        details_frame.pack(fill=X, pady=(0, 10))

        # Create a grid for frame details
        self.details_grid = ttk.Frame(details_frame)
        self.details_grid.pack(fill=X)

        # Resolution
        ttk.Label(self.details_grid, text="Resolution:").grid(row=0, column=0, sticky=W, padx=5)
        self.resolution_label = ttk.Label(self.details_grid, text="--")
        self.resolution_label.grid(row=0, column=1, sticky=W, padx=5)

        # FPS
        ttk.Label(self.details_grid, text="FPS:").grid(row=0, column=2, sticky=W, padx=5)
        self.fps_label = ttk.Label(self.details_grid, text="--")
        self.fps_label.grid(row=0, column=3, sticky=W, padx=5)

        # Frame count
        ttk.Label(self.details_grid, text="Frame:").grid(row=1, column=0, sticky=W, padx=5)
        self.frame_count_label = ttk.Label(self.details_grid, text="--")
        self.frame_count_label.grid(row=1, column=1, sticky=W, padx=5)

        # Total frames
        ttk.Label(self.details_grid, text="Total Frames:").grid(row=1, column=2, sticky=W, padx=5)
        self.total_frames_label = ttk.Label(self.details_grid, text="--")
        self.total_frames_label.grid(row=1, column=3, sticky=W, padx=5)

        # Prediction details section
        prediction_frame = ttk.LabelFrame(self.display_frame, text="Prediction Details", padding=10)
        prediction_frame.pack(fill=X, pady=(0, 10))

        # Create Treeview for prediction details
        pred_columns = ("Object", "Confidence", "Waste Category", "Box Coordinates")
        self.prediction_tree = ttk.Treeview(prediction_frame, columns=pred_columns, show="headings", height=4)
        
        # Configure columns
        self.prediction_tree.heading("Object", text="Object")
        self.prediction_tree.heading("Confidence", text="Confidence")
        self.prediction_tree.heading("Waste Category", text="Waste Category")
        self.prediction_tree.heading("Box Coordinates", text="Box Coordinates")
        
        self.prediction_tree.column("Object", width=100)
        self.prediction_tree.column("Confidence", width=100)
        self.prediction_tree.column("Waste Category", width=120)
        self.prediction_tree.column("Box Coordinates", width=200)
        
        # Add scrollbar
        pred_scrollbar = ttk.Scrollbar(prediction_frame, orient=VERTICAL, 
                                     command=self.prediction_tree.yview)
        self.prediction_tree.configure(yscrollcommand=pred_scrollbar.set)
        
        # Pack tree and scrollbar
        self.prediction_tree.pack(side=LEFT, fill=BOTH, expand=YES)
        pred_scrollbar.pack(side=RIGHT, fill=Y)

        # Right panel for detection text and history
        self.right_panel = ttk.LabelFrame(self.main_container, text="Detection Results", padding=10)
        self.right_panel.pack(side=RIGHT, fill=Y, padx=(10, 0))

        # Current detection text
        self.text_output = ScrolledText(self.right_panel, height=10, autohide=True)
        self.text_output.pack(fill=X, pady=(0, 10))

        # History section
        history_frame = ttk.LabelFrame(self.right_panel, text="Detection History", padding=10)
        history_frame.pack(fill=BOTH, expand=YES)

        # Create Treeview for history
        columns = ("Time", "Objects", "Confidence")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show="headings", 
                                       height=10)
        
        # Configure columns
        self.history_tree.heading("Time", text="Time")
        self.history_tree.heading("Objects", text="Objects")
        self.history_tree.heading("Confidence", text="Confidence")
        self.history_tree.column("Time", width=70)
        self.history_tree.column("Objects", width=150)
        self.history_tree.column("Confidence", width=70)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(history_frame, orient=VERTICAL, 
                                command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack tree and scrollbar
        self.history_tree.pack(side=LEFT, fill=BOTH, expand=YES)
        scrollbar.pack(side=RIGHT, fill=Y)

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = resize_for_display(rgb, width=800, height=480)
        from PIL import Image, ImageTk
        img = Image.fromarray(resized)
        imgtk = ImageTk.PhotoImage(img)
        self.frame_label.imgtk = imgtk
        self.frame_label.configure(image=imgtk)

        # Update frame details
        height, width = frame.shape[:2]
        self.resolution_label.configure(text=f"{width}x{height}")
        
        if self.controller.current_mode == "video":
            if self.controller.cap is not None:
                fps = self.controller.cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(self.controller.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = int(self.controller.cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                self.fps_label.configure(text=f"{fps:.2f}")
                self.frame_count_label.configure(text=str(current_frame))
                self.total_frames_label.configure(text=str(total_frames))
        elif self.controller.current_mode == "webcam":
            self.fps_label.configure(text="Live")
            self.frame_count_label.configure(text="--")
            self.total_frames_label.configure(text="--")
        else:
            self.fps_label.configure(text="--")
            self.frame_count_label.configure(text="--")
            self.total_frames_label.configure(text="--")

    def update_confidence(self, *args):
        self.detector.confidence_threshold = self.confidence_var.get()
        self.confidence_label.configure(text=f"Current: {self.confidence_var.get():.2f}")
        if self.controller.current_mode:
            self.controller.refresh_current_frame()

    def update_text(self, detected_objects):
        # Update current detection text
        self.text_output.delete(1.0, END)
        
        # Clear previous predictions
        for item in self.prediction_tree.get_children():
            self.prediction_tree.delete(item)
            
        if detected_objects:
            for obj in detected_objects:
                if obj['confidence'] >= self.confidence_var.get():
                    # Update text output with waste category
                    self.text_output.insert(END, f"{obj['class']}: {obj['confidence']:.2f} ({obj['waste_category']})\n")
                    
                    # Update prediction tree
                    box_coords = f"x1:{obj['box'][0]:.0f}, y1:{obj['box'][1]:.0f}, x2:{obj['box'][2]:.0f}, y2:{obj['box'][3]:.0f}"
                    self.prediction_tree.insert("", END, values=(
                        obj['class'],
                        f"{obj['confidence']:.2f}",
                        obj['waste_category'],
                        box_coords
                    ))
        else:
            self.text_output.insert(END, "No objects detected.")

        # Update history if in video or webcam mode
        if self.controller.current_mode in ["video", "webcam"]:
            current_time = datetime.now().strftime("%H:%M:%S")
            # Filter objects by confidence threshold
            filtered_objects = [obj for obj in detected_objects 
                              if obj['confidence'] >= self.confidence_var.get()]
            if filtered_objects:
                objects_str = ", ".join([f"{obj['class']}({obj['waste_category']})" for obj in filtered_objects])
                conf_str = ", ".join([f"{obj['confidence']:.2f}" for obj in filtered_objects])
                
                # Add to history
                self.history_tree.insert("", 0, values=(current_time, objects_str, conf_str))
                
                # Keep only last 100 entries
                if len(self.history_tree.get_children()) > 100:
                    self.history_tree.delete(self.history_tree.get_children()[-1])

    def detect_cameras(self):
        devices = []
        for i in range(5):  # check first 5 indices
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                devices.append(str(i))
            cap.release()
        return devices

    def select_image(self):
        self.stop()
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if path:
            self.controller.process_image(path)
            self.download_btn.configure(state=NORMAL)
            self.play_pause_btn.configure(state=DISABLED)

    def select_video(self):
        self.stop()
        path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov")])
        if path:
            self.controller.process_video(path)
            self.play_pause_btn.configure(state=NORMAL)
            self.download_btn.configure(state=DISABLED)

    def start_webcam(self):
        self.stop()
        device_index = int(self.device_var.get()) if self.device_var.get().isdigit() else 0
        self.controller.start_webcam(device_index)
        self.play_pause_btn.configure(state=DISABLED)
        self.download_btn.configure(state=DISABLED)

    def toggle_play_pause(self):
        self.controller.pause_video()
        if self.play_pause_btn.cget("text") == "Pause":
            self.play_pause_btn.configure(text="Play")
        else:
            self.play_pause_btn.configure(text="Pause")

    def stop(self):
        self.controller.stop()
        self.frame_label.configure(image='')
        self.download_btn.configure(state=DISABLED)
        self.play_pause_btn.configure(state=DISABLED)
        self.text_output.delete(1.0, END)
        
        # Clear history
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
            
        # Clear predictions
        for item in self.prediction_tree.get_children():
            self.prediction_tree.delete(item)
            
        # Reset frame details
        self.resolution_label.configure(text="--")
        self.fps_label.configure(text="--")
        self.frame_count_label.configure(text="--")
        self.total_frames_label.configure(text="--")

    def download_output(self):
        if self.controller.output_path:
            messagebox.showinfo("Download", f"Output saved to {self.controller.output_path}")
        else:
            messagebox.showwarning("Download", "No output file available")

    def on_close(self):
        self.stop()
        self.root.destroy()
