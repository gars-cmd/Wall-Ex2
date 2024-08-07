import cv2
import numpy as np
import pandas as pd
import os
import csv
import ast

import aruco_scanner_of_video

# Dictionary to store custom QR IDs
qr_id_map = {}
next_id = 0

#error treshold for the yaw angle
YAW_TRESHOLD = 0.5


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


def calculate_yaw_correction(old_yaw, current_yaw, treshold=YAW_TRESHOLD):
    delta = current_yaw - old_yaw
    if abs(delta) > treshold:
        if delta > 0:
            return "Yaw correction : Turn right"
        else:
            return "Yaw correction : Turn left"
    else:
        return "Yaw correction : Yaw angle correct"


def append_dict_to_csv(file_path, data_dict):
    try:
        file_exists = os.path.isfile(file_path)

        with open(file_path, 'a', newline='') as csvfile:
            fieldnames = ['FrameID'] + [f"Marker_{i}" for i in range(0, len(data_dict) - 1)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(data_dict)
    except:
        pass


def main(old_data, original_width, original_height):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    aruco_params = cv2.aruco.DetectorParameters()
    camera_matrix = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]])
    dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion
    marker_length = 0.14  # in meter

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)
    if not cap.isOpened():
        print("Error opening video capture")
        return

    saved_frame_corners = None

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
        detected_markers, data_dict = aruco_scanner_of_video.aruco_display(corners, ids, rejected, img, camera_matrix,
                                                                           dist_coeffs,
                                                                           marker_length, 0)
        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            if ids[0][0] in old_data:
                center_point = calculate_center_point(str(tuple(float(x) for x in corners[0][0][0])),
                                                      str(tuple(float(x) for x in corners[0][0][1])),
                                                      str(tuple(float(x) for x in corners[0][0][2])),
                                                      str(tuple(float(x) for x in corners[0][0][3])))

                cv2.arrowedLine(img, (int(old_data[ids[0][0]][0][0]), int(old_data[ids[0][0]][0][1])),(int(center_point[0]), int(center_point[1]))
                                , (0, 255, 0), 3,
                                tipLength=0.05)

                if data_dict:
                    new_distance = data_dict['QR 3D: dis']
                    old_distance = float(old_data[ids[0][0]][1])
                    delta = new_distance - old_distance
                    if delta > 0.2:
                        text = f"Distance correction : get closer"

                    elif delta < -0.2:
                        text = f"Distance correction : step back"
                    else:
                        text = "Distance correction : you're good"
                    cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

                    new_yaw = data_dict['QR 3D: yaw']
                    old_yaw = float(old_data[ids[0][0]][2])
                    yaw_correction = calculate_yaw_correction(old_yaw, new_yaw)
                    cv2.putText(img, yaw_correction, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.imshow('Live Video', img)
                continue
        cv2.imshow('Live Video', img)

        key = cv2.waitKey(1) & 0xFF
        if ids is not None:
            saved_frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            saved_frame_corners = corners
            save_frame_info(saved_frame_id, saved_frame_corners, ids)
            print("Frame information saved.")

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def calculate_center_point(corner1, corner2, corner3, corner4):
    # Parse the corner points from string to tuples
    corner1 = ast.literal_eval(corner1)
    corner2 = ast.literal_eval(corner2)
    corner3 = ast.literal_eval(corner3)
    corner4 = ast.literal_eval(corner4)

    # Calculate the center point
    center_x = (corner1[0] + corner2[0] + corner3[0] + corner4[0]) / 4
    center_y = (corner1[1] + corner2[1] + corner3[1] + corner4[1]) / 4
    return (center_x, center_y)


def csv_to_center_points(csv_file_path):
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        list_of_dicts = []

        for row in reader:
            center_point = calculate_center_point((str(eval(row['QR 2D'])[0])), (str(eval(row['QR 2D'])[1])),
                                                  (str(eval(row['QR 2D'])[2])),
                                                  (str(eval(row['QR 2D'])[3])))
            row['dis'] = row['QR 3D: dis']
            row['yaw'] = row['QR 3D: yaw']
            row['centerPoint'] = center_point
            list_of_dicts.append(row)

    return list_of_dicts


if __name__ == "__main__":
    cap = cv2.VideoCapture(fr'ChallengeB.mp4')
    # Get the original width and height of the video
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Example usage
    df = pd.read_csv(fr'output_scanner_CSV.csv')

    # Drop duplicates based on 'ID', keeping the first occurrence
    df_unique = df.drop_duplicates(subset=['ID'], keep='first')

    # Save the unique rows back to a new CSV file
    df_unique.to_csv('unique_data.csv', index=False)
    csv_file_path = 'unique_data.csv'  # Replace with your CSV file path
    data_with_centers = csv_to_center_points(csv_file_path)
    dict_centers = {}
    for row in data_with_centers:
        dict_centers[int(row['ID'])] = [row['centerPoint'], row['QR 3D: dis'], row['yaw']]
    main(dict_centers, original_width, original_height)
