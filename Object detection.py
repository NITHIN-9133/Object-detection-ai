import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Text, END, DISABLED, NORMAL, Scale, HORIZONTAL
from PIL import Image, ImageTk
import urllib.request
import time

class YOLOObjectDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Object Detection with YOLO")
        self.root.geometry("900x750")
        self.root.configure(bg="#f0f0f0")
        
        # YOLO model paths
        self.config_path = "yolov3.cfg"
        self.weights_path = "yolov3.weights"
        self.classes_path = "coco.names"
        
        # Create UI elements
        self.create_widgets()
        
        # Download YOLO files if not present
        self.download_yolo_files()
        
    def create_widgets(self):
        # Title label
        title_label = Label(self.root, text="Advanced Object Detection (YOLO)", font=("Arial", 20, "bold"), bg="#f0f0f0")
        title_label.pack(pady=20)
        
        # Upload button
        self.upload_button = Button(
            self.root, 
            text="Upload Image", 
            command=self.upload_image,
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white",
            padx=10,
            pady=5
        )
        self.upload_button.pack(pady=10)
        
        # Image display label
        self.image_label = Label(self.root, bg="#f0f0f0")
        self.image_label.pack(pady=10)
        
        # Confidence threshold slider
        slider_frame = tk.Frame(self.root, bg="#f0f0f0")
        slider_frame.pack(pady=5)
        
        confidence_label = Label(slider_frame, text="Confidence Threshold:", font=("Arial", 10), bg="#f0f0f0")
        confidence_label.pack(side=tk.LEFT, padx=5)
        
        self.confidence_slider = Scale(
            slider_frame, 
            from_=0.1, 
            to=1.0, 
            resolution=0.05, 
            orient=HORIZONTAL, 
            length=200,
            bg="#f0f0f0"
        )
        self.confidence_slider.set(0.5)  # Default value
        self.confidence_slider.pack(side=tk.LEFT, padx=5)
        
        # Detect button
        self.detect_button = Button(
            self.root, 
            text="Detect Objects", 
            command=self.detect_objects,
            font=("Arial", 12),
            bg="#2196F3",
            fg="white",
            padx=10,
            pady=5,
            state=DISABLED
        )
        self.detect_button.pack(pady=10)
        
        # Results display
        self.result_text = Text(self.root, height=12, width=80, font=("Arial", 10))
        self.result_text.pack(pady=10)
        
        # Status label
        self.status_label = Label(self.root, text="Status: Initializing...", font=("Arial", 10), bg="#f0f0f0")
        self.status_label.pack(pady=10)
    
    def download_yolo_files(self):
        """Download YOLO model files if not present"""
        self.status_label.config(text="Status: Checking for YOLO model files...")
        self.root.update()
        
        files_to_download = {
            "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
            "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
            "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights"
        }
        
        for file_name, url in files_to_download.items():
            if not os.path.exists(file_name):
                try:
                    self.status_label.config(text=f"Status: Downloading {file_name}...")
                    self.root.update()
                    
                    # For weights file (large), show progress
                    if file_name == "yolov3.weights":
                        self.result_text.insert(END, f"Downloading {file_name} (236MB)...\n")
                        self.result_text.insert(END, "This may take a few minutes. Please wait...\n")
                        self.root.update()
                    
                    urllib.request.urlretrieve(url, file_name)
                    
                    self.result_text.insert(END, f"Downloaded {file_name} successfully!\n")
                    self.root.update()
                except Exception as e:
                    self.status_label.config(text=f"Status: Error downloading {file_name}")
                    self.result_text.insert(END, f"Error downloading {file_name}: {str(e)}\n")
                    self.result_text.insert(END, "Please download YOLO files manually and place them in the same directory as this script.\n")
                    return False
        
        # Load the YOLO model
        try:
            self.status_label.config(text="Status: Loading YOLO model...")
            self.root.update()
            
            # Load the YOLO network
            self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
            
            # Load class names
            with open(self.classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
            
            self.status_label.config(text="Status: Ready! Upload an image to begin.")
            self.result_text.insert(END, "YOLO model loaded successfully!\n")
            self.result_text.insert(END, f"Model can detect {len(self.classes)} different object classes.\n")
            
            return True
        except Exception as e:
            self.status_label.config(text="Status: Error loading YOLO model")
            self.result_text.insert(END, f"Error loading YOLO model: {str(e)}\n")
            return False
    
    def upload_image(self):
        """Open a file dialog to select an image"""
        self.file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if self.file_path:
            # Update status
            self.status_label.config(text=f"Status: Image loaded from {os.path.basename(self.file_path)}")
            
            # Display the image
            self.display_image(self.file_path)
            
            # Enable detect button
            self.detect_button.config(state=NORMAL)
    
    def display_image(self, file_path):
        """Display the selected image"""
        # Open and resize image for display
        img = Image.open(file_path)
        img = self.resize_image(img, (500, 400))
        photo = ImageTk.PhotoImage(img)
        
        # Update image display
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference
        
        # Store the original image for processing
        self.original_img = cv2.imread(file_path)
    
    def resize_image(self, img, target_size):
        """Resize image while maintaining aspect ratio"""
        width, height = img.size
        ratio = min(target_size[0] / width, target_size[1] / height)
        new_size = (int(width * ratio), int(height * ratio))
        return img.resize(new_size, Image.LANCZOS)
    
    def detect_objects(self):
        """Detect objects in the uploaded image using YOLO"""
        if not hasattr(self, 'original_img'):
            self.status_label.config(text="Status: No image loaded!")
            return
        
        # Update status
        self.status_label.config(text="Status: Detecting objects...")
        self.result_text.delete(1.0, END)
        self.result_text.insert(END, "Processing image with YOLO...\n")
        self.root.update()
        
        try:
            start_time = time.time()
            
            # Get confidence threshold from slider
            confidence_threshold = self.confidence_slider.get()
            nms_threshold = 0.4  # Non-maximum suppression threshold
            
            # Prepare image for the network
            img = self.original_img.copy()
            height, width, channels = img.shape
            
            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)
            
            # Showing information on the screen
            class_ids = []
            confidences = []
            boxes = []
            
            # Process detections
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > confidence_threshold:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
            
            # Draw boxes and labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            detected_objects = {}
            
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = self.classes[class_ids[i]]
                    confidence = confidences[i]
                    
                    # Count objects by class
                    if label in detected_objects:
                        detected_objects[label] += 1
                    else:
                        detected_objects[label] = 1
                    
                    # Generate a random color for this class if not already assigned
                    color = (
                        int(((class_ids[i] * 100) % 255)),
                        int(((class_ids[i] * 150) % 255)),
                        int(((class_ids[i] * 200) % 255))
                    )
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw label background
                    text = f"{label}: {confidence:.2f}"
                    text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
                    cv2.rectangle(img, (x, y - 25), (x + text_size[0], y), color, -1)
                    
                    # Draw label text
                    cv2.putText(img, text, (x, y - 5), font, 0.5, (255, 255, 255), 2)
            
            # Calculate processing time
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Display results
            self.result_text.delete(1.0, END)
            self.result_text.insert(END, f"Detection completed in {processing_time:.2f} seconds\n\n")
            
            if detected_objects:
                self.result_text.insert(END, "Detected Objects:\n")
                for label, count in sorted(detected_objects.items(), key=lambda x: x[1], reverse=True):
                    self.result_text.insert(END, f"- {label}: {count}\n")
                
                total_objects = sum(detected_objects.values())
                self.result_text.insert(END, f"\nTotal objects detected: {total_objects}\n")
            else:
                self.result_text.insert(END, "No objects detected above the confidence threshold.\n")
                self.result_text.insert(END, "Try adjusting the confidence threshold slider.\n")
            
            # Convert back to PIL for display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img = self.resize_image(pil_img, (500, 400))
            photo = ImageTk.PhotoImage(pil_img)
            
            # Update image display
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference
            
            # Update status
            self.status_label.config(text=f"Status: Detection completed! Found {len(indexes)} objects.")
            
        except Exception as e:
            self.result_text.delete(1.0, END)
            self.result_text.insert(END, f"Error: {str(e)}")
            self.status_label.config(text="Status: Error in detection!")

def main():
    root = tk.Tk()
    app = YOLOObjectDetector(root)
    root.mainloop()

if __name__ == "__main__":
    main()