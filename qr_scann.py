import cv2
import time
import numpy as np

# Known width of the QR code (in meters or any other consistent units)
KNOWN_QR_CODE_WIDTH = 0.14  # Example: 14 cm



# Camera intrinsic parameters (these would typically be determined via calibration)
camera_matrix = np.array([[800, 0, 640],
                          [0, 800, 360],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion for simplicity

# Initialize the video capture with the first camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize the QR Code detector
detector = cv2.QRCodeDetector()





while cap.isOpened():
    success, img = cap.read()

    if not success:
        print("Failed to read from camera")
        break

    start = time.perf_counter()

    # Detect and decode the QR code
    value, points, _ = detector.detectAndDecode(img)

    if points is not None:
        print(f"QR code detected: {value}")
        points = points[0]  # Extract points from the list

        # Draw the bounding box
        for i in range(len(points)):
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[(i+1) % len(points)][0]), int(points[(i+1) % len(points)][1]))
            cv2.line(img, pt1, pt2, (0, 255, 0), 5)

        # Draw center point
        x_center = np.mean(points[:, 0])
        y_center = np.mean(points[:, 1])
        cv2.circle(img, (int(x_center), int(y_center)), 3, (0, 255, 0), 3)
        
        # Calculate the 3D position and yaw angle
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        obj_points = np.array([[0, 0, 0],
                               [KNOWN_QR_CODE_WIDTH, 0, 0],
                               [KNOWN_QR_CODE_WIDTH, KNOWN_QR_CODE_WIDTH, 0],
                               [0, KNOWN_QR_CODE_WIDTH, 0]], dtype=np.float32)
        
        success, rvec, tvec = cv2.solvePnP(obj_points, points, camera_matrix, dist_coeffs)
        if success:
            distance = np.linalg.norm(tvec)
            rmat, _ = cv2.Rodrigues(rvec)
            yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))

            cv2.putText(img, f"Distance: {distance:.2f} m", (30, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.putText(img, f"Yaw: {yaw:.2f} deg", (30, 170), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        else:
            print("Could not solve PnP")

        # Draw corner points coordinates
        for idx, (x, y) in enumerate(points):
            cv2.putText(img, f"P{idx+1} ({x:.2f}, {y:.2f})", (int(x), int(y-10)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    else:
        print("No QR code detected")

    end = time.perf_counter()
    total_time = end - start
    fps = 1 / total_time if total_time > 0 else 0

    cv2.putText(img, f'FPS: {int(fps)}', (30, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

