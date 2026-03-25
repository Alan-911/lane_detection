# Lane Detection Project (Professional Edition)

This repository provides an object-oriented, production-ready computer vision pipeline designed for robust real-time lane detection as outlined in the *Lane Detection Using Classical Computer Vision Techniques* project proposal by Yves Alain Iragena.

## Architecture & Features

This codebase deviates from standard procedural scripts by leveraging professional software engineering practices suitable for modular and extensive systems like ADAS:

- **Object-Oriented Design**: Encapsulates the entire processing logic inside a robust `LaneDetector` class.
- **Dependency Injection & Configurations**: Hyperparameters (e.g., Canny thresholds, Hough specifications, spatial footprints) are isolated in a cleanly typed `LaneDetectionConfig` dataclass payload.
- **Type Hinting**: All methods and logic flows enforce strict type hints (`typing`), improving readability and safety.
- **Structured Logging**: Bypasses arbitrary `print()` statements for a structured Python `logging` methodology, enabling timestamped debug and info telemetry.
- **Graceful Error Handling**: Fallbacks to original unstretched frames on calculation faults, along with structured argument parsing and path validations using `os` and `sys`.

## Methodology Pipeline

1. **Pre-processing**: Grayscale mapping and Gaussian noise gating.
2. **Feature Extraction**: High-fidelity gradient resolution via Canny edge detection.
3. **Spatial Masking**: Configured triangular Region of Interest (ROI) filtering.
4. **Line Detection**: Identifying structural segments via Probabilistic Hough Line Transform.
5. **Optimization**: Statistical slope/intercept segregation, averaging, and extrapolation into unified boundary markers.
6. **Telemetry & Output**: Real-time structured logging paired with `alpha`-blended geometric visual indicators.

## Environment Requirements

Dependencies are strictly cataloged. Install using virtual environments as necessary:

```sh
pip install -r requirements.txt
```

## Modular Usage

The primary interface for triggering the pipeline lies in `main.py`.

### For Dynamic Video Sources:
```sh
python main.py --type video --input /path/to/video.mp4
```

### For Static Image Sources:
```sh
python main.py --type image --input /path/to/image.jpg
```

*Note: You may further extend the CLI arguments seamlessly, such as tuning the Hough Line threshold:*
```sh
python main.py --type video --input demo.mp4 --hough_threshold 120
```
