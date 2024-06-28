import cv2
import numpy as np
import os
import csv

# Dictionary to store custom QR IDs
qr_id_map = {}
next_id = 0

def qr_display(corners, ids, rejected, image):
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
            print(f"QR Code ID: {markerID}")
            print(f"\tTop Left: {topLeft}")
            print(f"\tTop Right: {topRight}")
            print(f"\tBottom Right: {bottomRight}")
            print(f"\tBottom Left: {bottomLeft}")

            # Save the corners if you want to use them for calculations
            return (topLeft, topRight, bottomRight, bottomLeft)

def save_frame_info(frame_id, corners, ids):
    global next_id
    data_list = {"FrameID": frame_id}
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id not in qr_id_map:
            qr_id_map[marker_id] = next_id
            next_id += 1
        custom_id = qr_id_map[marker_id]
        marker_info = {
            "ID": custom_id,
            "QR 2D": f"[{corners[i][0][0]}, {corners[i][0][1]}, {corners[i][0][2]}, {corners[i][0][3]}]"
        }
        data_list[f"Marker_{custom_id}"] = marker_info

    file_path = 'saved_frame_info.csv'
    append_dict_to_csv(file_path, data_list)

def calculate_movements(saved_data, current_data):
    movements = []

    saved_x = saved_data[0][0]
    current_x = current_data[0][0]
    movement_x = current_x - saved_x

    if movement_x > 0:
        movements.append(f"Move right {movement_x} pixels")
    elif movement_x < 0:
        movements.append(f"Move left {abs(movement_x)} pixels")

    saved_y = saved_data[0][1]
    current_y = current_data[0][1]
    movement_y = current_y - saved_y

    if movement_y > 0:
        movements.append(f"Move down {movement_y} pixels")
    elif movement_y < 0:
        movements.append(f"Move up {abs(movement_y)} pixels")

    return movements

def append_dict_to_csv(file_path, data_dict):
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['FrameID'] + [f"Marker_{i}" for i in range(0, len(data_dict) - 1)]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data_dict)

def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video capture")
        return

    saved_frame_corners = None
    saved_frame_id = None

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)

        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(img, corners, ids, (0, 255, 0))

        cv2.imshow('Live Video', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            saved_frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            saved_frame_corners = corners
            save_frame_info(saved_frame_id, saved_frame_corners, ids)
            print("Frame information saved.")

        if key == ord('q'):
            break

        if saved_frame_corners is not None and len(corners) > 0:
            current_corners = corners[0][0]
            movements = calculate_movements(saved_frame_corners[0][0], current_corners)
            for move in movements:
                print(move)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()