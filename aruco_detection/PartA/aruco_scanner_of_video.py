import csv
import os
import cv2
import numpy as np

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


def aruco_display(corners, ids, rejected, image, camera_matrix, dist_coeffs, marker_length, frame_id):
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
            data_list = {"FrameID": frame_id}
            print(f"ArUco marker ID: {markerID}")
            data_list["ID"] = markerID
            corners_list = [topLeft, topRight, bottomRight, bottomLeft]
            data_list["QR 2D"] = corners_list

            # Estimate pose and print 3D information
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorner, marker_length, camera_matrix, dist_coeffs)
            # Distance from the camera to the marker(Distance Calculation)
            distance = np.linalg.norm(tvec)
            # Convert rotation vector to rotation matrix
            r, _ = cv2.Rodrigues(rvec)
            # Calculate yaw angle (rotation around y-axis)
            yaw = np.arctan2(r[1, 0], r[0, 0])
            # Normalize Yaw Angle:
            # Normalize Yaw Angle to -180 to 180 degrees
            yaw_degrees = np.degrees(yaw)
            yaw_degrees = (yaw_degrees + 180) % 360 - 180
            print(f"\tDistance: {distance:.2f} meters")
            data_list["QR 3D: dis"] = distance
            print(f"\tYaw angle: {yaw_degrees:.2f} degrees")
            data_list["QR 3D: yaw"] = yaw_degrees

            # Calculate pitch and roll angles (rotation around x and z axes respectively)
            pitch = np.arcsin(-r[2, 0])  # pitch angle
            roll = np.arctan2(r[2, 1], r[2, 2])  # roll angle
            pitch_degrees = np.degrees(pitch)
            roll_degrees = np.degrees(roll)
            print(f"\tPitch angle: {pitch_degrees:.2f} degrees")
            data_list["QR 3D: pitch"] = pitch_degrees
            print(f"\tRoll angle: {roll_degrees:.2f} degrees")
            data_list["QR 3D: roll"] = roll_degrees

            # Add distance, yaw, pitch, and roll to the image

            cv2.putText(image, f"Distance: {distance:.2f}m", (topLeft[0], topLeft[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, f"Yaw: {np.degrees(yaw):.2f} deg", (topLeft[0], topLeft[1] + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, f"Pitch: {pitch_degrees:.2f} deg", (topLeft[0], topLeft[1] + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, f"Roll: {roll_degrees:.2f} deg", (topLeft[0], topLeft[1] + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    if data_list:
        return image, data_list
    return image, None


def main(input_video):
    # Camera parameters (replace these with actual camera parameters)
    camera_matrix = np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]])
    dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion
    marker_length = 0.14  # in meter

    aruco_type = "DICT_4X4_100"
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    aruco_params = cv2.aruco.DetectorParameters()

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error opening video file")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Output video settings
    output_file = 'output_scanner_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    frame_id = 0
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        frame_id += 1

        corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
        detected_markers, data_dict = aruco_display(corners, ids, rejected, img, camera_matrix, dist_coeffs,
                                                    marker_length, frame_id)
        if data_dict:
            append_dict_to_csv('output_scanner_CSV.csv', data_dict)

        out.write(detected_markers)  # Write frame to output video
        cv2.imshow("Video", detected_markers)

        key = cv2.waitKey(5) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release()


def append_dict_to_csv(file_path, data_dict):
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data_dict)


if __name__ == "__main__":
    input_video = r"C:\Users\User\Desktop\לירון\AutoRobots\Ex2PartAandB\Wall-Ex2\aruco_detection\Mp4Tests\ChallengeB.mp4"
    main(input_video)