<p align="center">
  <img src="https://learnopencv.com/wp-content/uploads/2020/03/aruco-markers-examples.jpg" alt="Logo" width="450" height="300" align="right">
</p>

# ArUco Scanner Project

Welcome to the ArUco Scanner Project! This repository contains scripts to detect ArUco markers in video files and real-time camera feeds. The project uses OpenCV and NumPy for marker detection and distance calculation.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
    - [Running ArUco Scanner on a Video File](#running-aruco-scanner-on-a-video-file)
    - [Running ArUco Scanner Live](#running-aruco-scanner-live)
- [Modifying ArUco Tag Parameters](#modifying-aruco-tag-parameters)

## Project Description

This project provides tools to detect and measure distances to ArUco markers in both prerecorded video and real-time camera feeds. By default, the project uses a 4x4 ArUco tag dictionary with 100 unique markers, but this can be customized to suit different requirements.

## Installation

To run the project, you'll need Python, as well as the most recent versions of OpenCV and NumPy. 

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/aruco_scanner_project.git
    cd aruco_scanner_project
    ```

2. Install the required dependencies:
    ```sh
    pip install opencv-python numpy
    pip install opencv-python==4.6.0.66
    python3 -m pip install opencv-contrib-python
    ```

## Usage

### Running ArUco Scanner on a Video File

To detect ArUco markers in a video file, use the `aruco_scanner.py` script with the following command:
```sh
python aruco_scanner.py input.mp4
```
Replace `input.mp4` with the path to your video file.
![image](https://github.com/gars-cmd/Wall-Ex2/assets/75199660/ddc980f5-39af-4006-8ca2-56034cd82123)



### Running ArUco Scanner Live

To detect ArUco markers from a live camera feed, use the `aruco_scanner_live.py` script with the following command:
```sh
python aruco_scanner_live.py 0
```
If you have multiple cameras, replace `0` with the appropriate camera index.

## Modifying ArUco Tag Parameters

To ensure precise distance calculations, you may need to modify the size and type of the ArUco tag. This can be done directly in the script files.

### Editing the ArUco Dictionary and Tag Size

1. **Change ArUco Dictionary**: Open the appropriate script file (`aruco_scanner.py` or `aruco_scanner_live.py`) and modify in the main function the `aruco_type` variable. Modify it to use a different dictionary if needed.
    ```python
    aruco_type = "DICT_4X4_100"
    # Replace with another dictionary, e.g., "DICT_6X6_250"
    ```

2. **Set the ArUco Tag Size**: Define the size of the ArUco tag in meters (or any consistent unit) for distance calculation:
    ```python
    tag_size = 0.05  # Default size in meters
    # Change this value to match the physical size of your tags
    ```

By adjusting these parameters, you can fine-tune the accuracy of the marker detection system.

