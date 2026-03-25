import numpy as np
import pytest
from lane_detector import LaneDetector, LaneDetectionConfig

def test_lane_detector_initialization():
    """Verify detector adopts overridden configs seamlessly."""
    config = LaneDetectionConfig(hough_threshold=50)
    detector = LaneDetector(config=config)
    assert detector.config.hough_threshold == 50

def test_process_empty_frame():
    """Ensure pipeline survives edge cases (like pitch black noise)."""
    detector = LaneDetector()
    # Create a 720p TuSimple-like black canvas
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    result = detector.process_frame(dummy_frame)
    
    # Validation
    assert result.shape == dummy_frame.shape
    assert result.dtype == dummy_frame.dtype

def test_offset_calculation():
    """Test the physical pixel-to-meter matrix scaling logic."""
    detector = LaneDetector()
    # Simulating a perfectly center lane where image width is 1280
    # Left line intersects bottom at x=440
    # Right line intersects bottom at x=840
    # Center = (440 + 840) / 2 = 640
    # True image center = 1280 / 2 = 640. Offset = 0
    left_line = np.array([440, 720, 600, 400])
    right_line = np.array([840, 720, 680, 400])
    
    offset_m = detector._calculate_offset(1280, left_line, right_line)
    
    assert offset_m is not None
    assert abs(offset_m) < 0.001  # Approximately zero
    
def test_offset_calculation_drift_left():
    """Test calculation when vehicle drifts left off center."""
    detector = LaneDetector()
    left_line = np.array([540, 720, 600, 400])
    right_line = np.array([940, 720, 680, 400]) # Lane center is 740, img center is 640
    
    offset_m = detector._calculate_offset(1280, left_line, right_line)
    
    # Center shifted right (740) relative to image center (640)
    # Means vehicle drifted left logically depending on origin coordinate
    # (640 - 740) * scale = -100 * (3.7/700) = -0.528m
    assert offset_m is not None
    assert offset_m < 0 
