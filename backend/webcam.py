"""
Standalone webcam object detection script
Run this for local real-time detection without API
"""
import logging
import time
import cv2
from backend.model import get_model_manager
from backend.config import settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run live webcam detection"""
    logger.info("Initializing webcam detection...")
    
    # Load model
    try:
        model_manager = get_model_manager()
        logger.info(f"Model loaded: {model_manager.get_model_info()['model_name']}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    logger.info("Webcam opened successfully. Press ESC to exit.")
    
    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    # Frame skipping for better performance
    frame_skip = 3  # Process every 3rd frame
    frame_counter = 0
    last_annotated_frame = None
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                logger.warning("Failed to read frame from webcam")
                break
            
            frame_counter += 1
            process_frame = (frame_counter % frame_skip == 0)
            
            # Run detection only on selected frames
            if process_frame:
                # Run detection
                results = model_manager.predict(frame)
                annotated_frame = results[0].plot()
                last_annotated_frame = annotated_frame.copy()  # Store for later
                
                # Calculate FPS
                fps_frame_count += 1
                if fps_frame_count >= 10:
                    fps_end_time = time.time()
                    fps = fps_frame_count / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                    fps_frame_count = 0
                    logger.info(f"FPS: {fps:.1f}")
                
                # Display FPS on frame
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Display detection count
                detection_count = len(results[0].boxes)
                cv2.putText(
                    annotated_frame,
                    f"Objects: {detection_count}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            else:
                # Use last detection result for smooth display
                if last_annotated_frame is not None:
                    annotated_frame = last_annotated_frame.copy()
                    cv2.putText(
                        annotated_frame,
                        "Processing...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2
                    )
                else:
                    # Fallback to current frame if no detection yet
                    annotated_frame = frame.copy()
                    cv2.putText(
                        annotated_frame,
                        "Processing...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2
                    )
            
            cv2.imshow("YOLO Live Detection - Press ESC to exit", annotated_frame)
            
            # Reduce jitter by delaying a tiny bit in the display loop
            key = cv2.waitKey(1)
            if key == 27:
                logger.info("ESC pressed, exiting...")
                break

            # Slight sleep reduces CPU spikes and smooths frame cadence
            time.sleep(0.004)  # 4ms
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam detection stopped")


if __name__ == "__main__":
    main()
