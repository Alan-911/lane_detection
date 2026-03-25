import cv2
import argparse
import sys
import logging
import os
from lane_detector import LaneDetector, LaneDetectionConfig

# Configure application-wide logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_image(detector: LaneDetector, source_path: str):
    """Processes a single static image."""
    logger.info(f"Loading image from {source_path}")
    if not os.path.exists(source_path):
        logger.error(f"Image path does not exist: {source_path}")
        sys.exit(1)

    image = cv2.imread(source_path)
    if image is None:
        logger.error("Failed to load image. Ensure it is a valid visual format.")
        sys.exit(1)

    logger.info("Running lane detection pipeline...")
    result_image = detector.process_frame(image)
    
    cv2.imshow('Lane Detection Result', result_image)
    logger.info("Displaying result. Press any key in the window to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(detector: LaneDetector, source_path: str):
    """Processes a continuous video stream."""
    logger.info(f"Loading video from {source_path}")
    if not os.path.exists(source_path):
        logger.error(f"Video path does not exist: {source_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        logger.error("Failed to open video stream. Ensure the codec is supported.")
        sys.exit(1)

    logger.info("Initializing video processing. Press 'q' to quit at any time.")
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream reached.")
            break
            
        frame_count += 1
        result_frame = detector.process_frame(frame)
        cv2.imshow('Lane Detection Pipeline', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Processing interrupted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Video processing completed. Total frames processed: {frame_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Professional Lane Detection Pipeline built for robust local benchmarking."
    )
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help="Path to the input video or image source.")
    parser.add_argument('-t', '--type', type=str, choices=['image', 'video'], default='video', 
                        help="Specify if the input is an 'image' or 'video' file.")
    parser.add_argument('--hough_threshold', type=int, default=100, 
                        help="Hough transform intersection threshold override.")
    
    args = parser.parse_args()

    # Initialize configuration (with optional overrides via CLI)
    config = LaneDetectionConfig(hough_threshold=args.hough_threshold)
    
    # Initialize the robust detector object
    detector = LaneDetector(config=config)

    try:
        if args.type == 'image':
            process_image(detector, args.input)
        elif args.type == 'video':
            process_video(detector, args.input)
        else:
            logger.error("Unsupported file type specified.")
    except KeyboardInterrupt:
        logger.info("Execution cancelled by user.")
    except Exception as e:
        logger.exception("An unexpected failure occurred in the main process:")
        sys.exit(1)

if __name__ == '__main__':
    main()
