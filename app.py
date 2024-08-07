from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import gradio as gr
import cv2

# Create directories to save uploaded and result files
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Load the model with the saved weights on CPU
model = YOLO('best.pt').to('cpu')

# Gradio Interface Function
def gradio_interface(input_type, input_data):
    if input_type == "Webcam":
        # Initialize video capture
        cap = cv2.VideoCapture(input_data)
        frames = []
        processed_results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to PIL format
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Perform inference
            results = model.predict(source=image)
            
            # Get the annotated image with bounding boxes
            annotated_image = results[0].plot()
            
            # Append the annotated frame to the list
            frames.append(annotated_image)
            
            # Process results for Gradio display
            for result in results:
                boxes = result.boxes.xyxy.tolist()
                classes = result.boxes.cls.tolist()
                confs = result.boxes.conf.tolist()
                
                for box, cls, conf in zip(boxes, classes, confs):
                    processed_results.append({
                        "box": box,
                        "class": int(cls),
                        "confidence": float(conf)
                    })

        cap.release()
        
        # Convert frames to a video
        height, width, layers = frames[0].shape
        video_path = os.path.join(RESULT_DIR, "result.mp4")
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        
        return video_path, None, processed_results

    elif input_type == "Image":
        # Perform inference
        results = model.predict(source=input_data)
        
        # Get the annotated image with bounding boxes
        annotated_image = results[0].plot()
        
        # Save the annotated image
        result_filename = "result.png"
        result_path = os.path.join(RESULT_DIR, result_filename)
        Image.fromarray(annotated_image).save(result_path)
        
        # Process results for Gradio display
        processed_results = []
        for result in results:
            boxes = result.boxes.xyxy.tolist()
            classes = result.boxes.cls.tolist()
            confs = result.boxes.conf.tolist()
            
            for box, cls, conf in zip(boxes, classes, confs):
                processed_results.append({
                    "box": box,
                    "class": int(cls),
                    "confidence": float(conf)
                })
        
        return None, result_path, processed_results

# Define the Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# Object Detection with YOLOv8")
    
    with gr.Row():
        input_type = gr.Radio(["Webcam", "Image"], label="Input Type", value="Image")
    
    with gr.Row():
        with gr.Column():
            webcam_input = gr.Video(sources="webcam", label="Webcam Video", visible=False)
            image_input = gr.Image(type="numpy", label="Image")
        
        with gr.Column():
            video_output = gr.Video(label="Processed Video")
            image_output = gr.Image(type="filepath", label="Processed Image")
    
    json_output = gr.JSON(label="Detection Results")
    
    submit_button = gr.Button("Submit")
    
    def update_input_type(choice):
        return {
            webcam_input: gr.update(visible=choice == "Webcam"),
            image_input: gr.update(visible=choice == "Image")
        }
    
    input_type.change(update_input_type, input_type, [webcam_input, image_input])
    
    def process_input(input_type, webcam, image):
        if input_type == "Webcam":
            return gradio_interface("Webcam", webcam)
        else:
            return gradio_interface("Image", image)
    
    submit_button.click(
        process_input,
        inputs=[input_type, webcam_input, image_input],
        outputs=[video_output, image_output, json_output]
    )

# Launch the Gradio interface
interface.launch()