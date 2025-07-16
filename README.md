# CubeEye Hand Detection and Point Cloud Visualization

This repository contains a Python script for hand detection and point cloud visualization using the CubeEye SDK, Open3D, and YOLO.

## Features

- Real-time hand detection using YOLO.
- Point cloud visualization with Open3D.
- Depth and amplitude image processing.
- Dataset recording for detected hands, including depth, amplitude, and point cloud data.

## Requirements

To run this script, you need the following:

1. **CubeEye SDK**: Ensure the CubeEye SDK is installed and running on your PC.
2. **Python 3.8 or higher**: The script is tested with Python 3.8.
3. **Dependencies**: Install the required Python libraries using the command below.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/RAPTOR-sr/CubeEye_Hand_detection_Script
   cd cubeeye-hand-detection
   ```
2. Install the required Python libraries:
   ```bash
   pip install numpy opencv-python open3d ultralytics pywin32
   ```

## Usage

1. Connect your CubeEye device to the PC.
2. Run the hand detection script:
   ```bash
   python hand_detection.py
   ```
3. Follow the on-screen instructions to perform hand detection and visualize the point cloud.

## Dataset Recording

To record a dataset of detected hands:

1. Ensure the CubeEye device is properly connected and calibrated.
2. Run the dataset recording script:
   ```bash
   python record_dataset.py
   ```
3. The recorded dataset will be saved in the `dataset/` directory by default.

## Dataset Directory Structure

The recorded dataset follows this directory structure:

```
dataset/
├── depth/
│   ├── hand_000001.png
│   ├── hand_000002.png
│   └── ...
├── amplitude/
│   ├── hand_000001.png
│   ├── hand_000002.png
│   └── ...
├── pointcloud/
│   ├── hand_000001.ply
│   ├── hand_000002.ply
│   └── ...
└── metadata/
    ├── hand_000001.json
    ├── hand_000002.json
    └── ...
```

Where:
- `depth/`: Contains depth images of detected hands in PNG format
- `amplitude/`: Contains amplitude/intensity images in PNG format
- `pointcloud/`: Contains 3D point cloud data in PLY format
- `metadata/`: Contains JSON files with detection metadata (bounding boxes, timestamps, etc.)

## Troubleshooting

- If you encounter any issues, ensure that the CubeEye SDK is correctly installed and the device is properly connected.
- For dependency-related issues, ensure all required Python libraries are installed.

## Acknowledgments

- **CubeEye SDK**: For providing the SDK and tools for CubeEye device integration.
- **Open3D**: For the point cloud visualization library.
- **YOLO**: For the real-time object detection model.

