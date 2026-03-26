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
    
    # ROI parameters - Tuned for TuSimple (1280x720)
    roi_bottom_width_pct: float = 1.0  # Full width at bottom
    roi_top_width_pct: float = 0.1     # Narrow crop near horizon
    roi_height_pct: float = 0.6        # Height horizon
    
    # Hough Transform parameters - Tuned for TuSimple dashes
    hough_rho: int = 2
    hough_theta: float = np.pi / 180
    hough_threshold: int = 50          # Lowered to capture faded TuSimple lanes
    hough_min_line_len: int = 40
    hough_max_line_gap: int = 100      # Increased significantly to bridge long dashed gaps
    
    # Visualization parameters
    line_color: Tuple[int, int, int] = (255, 0, 0) # Blue lanes for high contrast
    line_thickness: int = 10
    overlay_alpha: float = 0.8
    line_alpha: float = 1.0
    
    # Dashboard parameters
    xm_per_pix: float = 3.7 / 700      # Meters per pixel approx for US Highway


class LaneDetector:
    """
    LaneDetector implements a classical computer vision pipeline
    for detecting driving lanes in images and video streams.
    """
    def __init__(self, config: Optional[LaneDetectionConfig] = None):
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

    def _average_slope_intercept(self, image: np.ndarray, lines: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Groups detected segments into left and right lanes, averages them, and returns (left_line, right_line)."""
        left_fit = []
        right_fit = []

        if lines is None:
            return None, None

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = parameters[0], parameters[1]

            if slope < -0.3: # Filter out horizontal noise
                left_fit.append((slope, intercept))
            elif slope > 0.3:
                right_fit.append((slope, intercept))

        left_line = None
        right_line = None

        if left_fit:
            left_avg = np.average(left_fit, axis=0)
            left_line = self._make_coordinates(image, left_avg)

        if right_fit:
            right_avg = np.average(right_fit, axis=0)
            right_line = self._make_coordinates(image, right_avg)

        return left_line, right_line

    def _calculate_offset(self, width: int, left_line: Optional[np.ndarray], right_line: Optional[np.ndarray]) -> Optional[float]:
        """Calculates the physical vehicle offset from the lane center in meters."""
        if left_line is None or right_line is None:
            return None
            
        # Extract the bottom x coordinates (x1)
        left_x_bottom = left_line[0]
        right_x_bottom = right_line[0]
        
        lane_center_px = (left_x_bottom + right_x_bottom) / 2.0
        image_center_px = width / 2.0
        
        # Offset in pixels: positive means car is right of the center
        offset_pixels = image_center_px - lane_center_px
        
        return offset_pixels * self.config.xm_per_pix

    def _draw_dashboard(self, image: np.ndarray, offset: Optional[float]) -> np.ndarray:
        """Embeds a telemetry dashboard on the top-right corner of the frame."""
        overlay = np.copy(image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw background rectangle for dashboard
        cv2.rectangle(overlay, (20, 20), (600, 100), (0, 0, 0), -1)
        
        if offset is not None:
            direction = "Left" if offset < 0 else "Right"
            text = f"Vehicle Offset: {abs(offset):.2f}m {direction}"
            color = (0, 255, 0) if abs(offset) < 0.5 else (0, 0, 255) # Red if drifted > 0.5m
        else:
            text = "Vehicle Offset: Unavailable"
            color = (255, 255, 255)
            
        cv2.putText(overlay, text, (40, 70), font, 1, color, 2, cv2.LINE_AA)
        return cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

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

    def _display_lines(self, image: np.ndarray, left_line: Optional[np.ndarray], right_line: Optional[np.ndarray], is_warning: bool = False) -> np.ndarray:
        """Draws the extrapolated lines onto a blank image matching the input dimensions."""
        line_image = np.zeros_like(image)
        lines = [l for l in (left_line, right_line) if l is not None]
        
        # Override to strict Red (0, 0, 255 in BGR) if warning is active, else use standard config color
        render_color = (0, 0, 255) if is_warning else self.config.line_color
        
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), render_color, self.config.line_thickness)
        return line_image

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Main pipeline method that processes a single BGR frame."""
        if frame is None or not isinstance(frame, np.ndarray):
            logger.error("Invalid frame passed to process_frame.")
            return frame

        try:
            lane_image = np.copy(frame)
            height, width = lane_image.shape[:2]
            
            # Feature Extraction
            canny_image = self._canny_edge_detection(lane_image)
            cropped_image = self._region_of_interest(canny_image)
            
            # Line Detection
            lines = cv2.HoughLinesP(
                cropped_image,
                rho=self.config.hough_rho,
                theta=self.config.hough_theta,
                threshold=self.config.hough_threshold,
                lines=np.array([]),
                minLineLength=self.config.hough_min_line_len,
                maxLineGap=self.config.hough_max_line_gap
            )
            
            # Line Optimization and Offset Analytics
            left_line, right_line = self._average_slope_intercept(lane_image, lines)
            offset_meters = self._calculate_offset(width, left_line, right_line)
            
            # Determine if vehicle has drifted dangerously out of lane (> 0.5m offset)
            is_drifting = offset_meters is not None and abs(offset_meters) > 0.5
            
            # Outputs with Dynamic Red-line warning triggering
            line_overlay = self._display_lines(lane_image, left_line, right_line, is_warning=is_drifting)
            blended_lanes = cv2.addWeighted(
                lane_image, 
                self.config.overlay_alpha, 
                line_overlay, 
                self.config.line_alpha, 
                1
            )
            
            # Embed Dashboard
            return self._draw_dashboard(blended_lanes, offset_meters)
            
        except Exception as e:
            logger.error(f"Error during frame processing: {e}")
            return frame
