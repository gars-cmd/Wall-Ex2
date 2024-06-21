import csv
import os

import cv2
import numpy as np
import argparse

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


def aruco_display(corners, ids, rejected, image, camera_matrix, dist_coeffs, marker_length):
    data_list = None
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Print the corner coordinates
            data_list = {}
            print(f"ArUco marker ID: {markerID}")
            data_list["ID"] = markerID
            print(f"\tTop Left: {topLeft}")
            data_list["topLeft"] = topLeft
            print(f"\tTop Right: {topRight}")
            data_list["topRight"] = topRight
            print(f"\tBottom Right: {bottomRight}")
            data_list["bottomRight"] = bottomRight
            print(f"\tBottom Left: {bottomLeft}")
            data_list["bottomLeft"] = bottomLeft

            # Estimate pose and print 3D information
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorner, marker_length, camera_matrix, dist_coeffs)
            # Distance from the camera to the marker
            distance = np.linalg.norm(tvec)
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            # Calculate yaw angle (rotation around y-axis)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            print(f"\tDistance: {distance:.2f} meters")
            data_list["distance"] = distance
            print(f"\tYaw angle: {np.degrees(yaw):.2f} degrees")
            data_list["yaw"] = np.degrees(yaw)

    if data_list:
        return image, data_list
    return image, None  # Ensure the image is always returned


def main(camera_index):
    # Camera parameters (replace these with actual camera parameters)
    camera_matrix = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]])
    dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion
    # WARNING: DO NOT FORGET TO UPDATE THE SIZE OF THE MARKER LENGTH TO MAKE THE DISTANCE CALCULATION REAL
    marker_length = 0.015  # Marker length in meters

    aruco_type = "DICT_4X4_100"
    aruco_Dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    arucoParams = cv2.aruco.DetectorParameters()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error opening video capture")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        h, w, _ = img.shape

        width = 1000
        height = int(width * (h / w))
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

        corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_Dict, parameters=arucoParams)
        detected_markers, data_dict = aruco_display(corners, ids, rejected, img, camera_matrix, dist_coeffs,
                                                    marker_length)
        if data_dict:
            append_dict_to_csv('output.csv', data_dict)

        cv2.imshow("Live Video", detected_markers)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


def append_dict_to_csv(file_path, data_dict):
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())

        # If the file does not exist, write the header
        if not file_exists:
            writer.writeheader()

        # Write the dictionary as a row in the CSV file
        writer.writerow(data_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArUco Marker Detection from Live Camera")
    parser.add_argument("camera_index", type=int, help="Index of the camera")
    args = parser.parse_args()

    main(args.camera_index)
