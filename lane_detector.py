import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

@dataclass
class LaneDetectionConfig:
    """Configuration parameters for the Lane Detection pipeline."""
    # Pre-processing parameters
    blur_kernel_size: Tuple[int, int] = (5, 5)
    
    # Canny parameters
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    
    # ROI parameters
    roi_bottom_width_pct: float = 0.8  # Bottom width relative to frame width
    roi_top_width_pct: float = 0.1     # Top width relative to frame width
    roi_height_pct: float = 0.6        # Height of the ROI from the bottom
    
    # Hough Transform parameters
    hough_rho: int = 2
    hough_theta: float = np.pi / 180
    hough_threshold: int = 100
    hough_min_line_len: int = 40
    hough_max_line_gap: int = 5
    
    # Visualization parameters
    line_color: Tuple[int, int, int] = (0, 255, 0)
    line_thickness: int = 10
    overlay_alpha: float = 0.8
    line_alpha: float = 1.0


class LaneDetector:
    """
    LaneDetector implements a classical computer vision pipeline
    for detecting driving lanes in images and video streams.
    """
    def __init__(self, config: Optional[LaneDetectionConfig] = None):
        """
        Initializes the LaneDetector with the given configuration.
        """
        self.config = config or LaneDetectionConfig()

    def _make_coordinates(self, image: np.ndarray, line_parameters: Tuple[float, float]) -> np.ndarray:
        """Calculates pixel coordinates for a lane line given its slope and intercept."""
        slope, intercept = line_parameters
        height = image.shape[0]
        y1 = height
        y2 = int(height * self.config.roi_height_pct)

        # Prevent division by zero
        if abs(slope) < 1e-4:
            slope = 1e-4 if slope >= 0 else -1e-4

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    def _average_slope_intercept(self, image: np.ndarray, lines: Optional[np.ndarray]) -> List[np.ndarray]:
        """Groups detected segments into left and right lanes, averages them, and extrapolates."""
        left_fit = []
        right_fit = []

        if lines is None:
            return []

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = parameters[0], parameters[1]

            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        averaged_lines = []
        if left_fit:
            left_avg = np.average(left_fit, axis=0)
            averaged_lines.append(self._make_coordinates(image, left_avg))

        if right_fit:
            right_avg = np.average(right_fit, axis=0)
            averaged_lines.append(self._make_coordinates(image, right_avg))

        return averaged_lines

    def _canny_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """Applies grayscale conversion, Gaussian blur, and Canny edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self.config.blur_kernel_size, 0)
        return cv2.Canny(blur, self.config.canny_low_threshold, self.config.canny_high_threshold)

    def _region_of_interest(self, canny_image: np.ndarray) -> np.ndarray:
        """Creates a spatial mask representing the region of interest and applies it."""
        height, width = canny_image.shape
        
        bottom_margin = (1.0 - self.config.roi_bottom_width_pct) / 2.0
        top_margin = (1.0 - self.config.roi_top_width_pct) / 2.0
        
        pt1 = (int(width * bottom_margin), height)
        pt2 = (int(width * (1.0 - bottom_margin)), height)
        pt3 = (int(width * (1.0 - top_margin)), int(height * self.config.roi_height_pct))
        pt4 = (int(width * top_margin), int(height * self.config.roi_height_pct))
        
        polygons = np.array([[pt1, pt2, pt3, pt4]], np.int32)
        mask = np.zeros_like(canny_image)
        cv2.fillPoly(mask, polygons, 255)
        
        return cv2.bitwise_and(canny_image, mask)

    def _display_lines(self, image: np.ndarray, lines: List[np.ndarray]) -> np.ndarray:
        """Draws the extrapolated lines onto a blank image matching the input dimensions."""
        line_image = np.zeros_like(image)
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), self.config.line_color, self.config.line_thickness)
        return line_image

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Main pipeline method that processes a single BGR frame."""
        if frame is None or not isinstance(frame, np.ndarray):
            logger.error("Invalid frame passed to process_frame.")
            return frame

        try:
            lane_image = np.copy(frame)
            
            # 1 & 2: Pre-Processing and Feature Extraction
            canny_image = self._canny_edge_detection(lane_image)
            
            # 3: Spatial Masking
            cropped_image = self._region_of_interest(canny_image)
            
            # 4: Line Detection
            lines = cv2.HoughLinesP(
                cropped_image,
                rho=self.config.hough_rho,
                theta=self.config.hough_theta,
                threshold=self.config.hough_threshold,
                lines=np.array([]),
                minLineLength=self.config.hough_min_line_len,
                maxLineGap=self.config.hough_max_line_gap
            )
            
            # 5: Optimization
            averaged_lines = self._average_slope_intercept(lane_image, lines)
            
            # 6: Output Overlay
            line_overlay = self._display_lines(lane_image, averaged_lines)
            return cv2.addWeighted(
                lane_image, 
                self.config.overlay_alpha, 
                line_overlay, 
                self.config.line_alpha, 
                1
            )
        except Exception as e:
            logger.error(f"Error during frame processing: {e}")
            return frame
