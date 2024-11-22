import cv2
from ultralytics import YOLO  # For YOLOv8
import time
import os


# Function to run object detection using YOLO
def run_yolo_detection(video_path, model_path):
    try:
        print("Loading YOLO model...")
        # Load YOLO model (YOLOv5 or YOLOv8)
        model = YOLO(model_path)

        print("Starting object detection...")
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        total_time = 0

        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_path}")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Start timing
            start_time = time.time()

            # Perform detection
            results = model.predict(source=frame, save=False, verbose=False)

            # End timing
            end_time = time.time()
            frame_time = end_time - start_time
            total_time += frame_time
            frame_count += 1

            # Display results on the frame
            annotated_frame = results[0].plot()  # Annotates detected objects
            cv2.imshow("YOLO Object Detection", annotated_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            print(f"Frame {frame_count}: Processing time = {frame_time:.3f} seconds")

        cap.release()
        cv2.destroyAllWindows()
        print(f"Total frames processed: {frame_count}")
        print(f"Average processing time per frame: {total_time / frame_count:.3f} seconds")

    except Exception as e:
        print(f"Error in YOLO detection: {e}")


# Main function
def main():
    video_path = input("Enter the path to your MP4 video file: ")
    model_path = "yolov8n.pt"  # YOLOv8 nano pre-trained model

    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        return

    # Check if YOLO model file exists
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found. Downloading YOLOv8 model...")
        from ultralytics.yolo.utils import download
        download(model_path)
        print("Model downloaded successfully.")

    # Run YOLO object detection
    run_yolo_detection(video_path, model_path)


# Entry point
if __name__ == "__main__":
    main()
