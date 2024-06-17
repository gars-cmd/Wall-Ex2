import cv2
import numpy as np
import time

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

# Predefined QR code sizes in meters
QR_CODE_SIZES = [0.14, 0.07, 0.03]

def order_points(pts):
    assert pts.shape[0] == 4, "Expected exactly 4 points for QR code edges detection."
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def compute_distance(obj_points, img_points):
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
    if success:
        distance = np.linalg.norm(tvec)
        rmat, _ = cv2.Rodrigues(rvec)
        yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
        return distance, yaw
    else:
        print("Could not solve PnP")
        return None, None

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
        points = points[0]
        points = order_points(points)

        # Draw the bounding box and center point
        for i in range(len(points)):
            pt1 = (int(points[i][0]), int(points[i][1]))
            pt2 = (int(points[(i+1) % len(points)][0]), int(points[(i+1) % len(points)][1]))
            cv2.line(img, pt1, pt2, (0, 255, 0), 5)
            
            # Display 2D coordinates of the corners
            cv2.putText(img, f'({pt1[0]}, {pt1[1]})', pt1, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        x_center = np.mean(points[:, 0])
        y_center = np.mean(points[:, 1])
        cv2.circle(img, (int(x_center), int(y_center)), 3, (0, 255, 0), 3)

        best_distance = None
        best_yaw = None

        for size in QR_CODE_SIZES:
            obj_points = np.array([[0, 0, 0],
                                   [size, 0, 0],
                                   [size, size, 0],
                                   [0, size, 0]], dtype=np.float32)
            distance, yaw = compute_distance(obj_points, points)
            if distance is not None:
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_yaw = yaw
        
        if best_distance is not None:
            cv2.putText(img, f"Distance: {best_distance:.2f} m", (30, 120), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.putText(img, f"Yaw: {best_yaw:.2f} deg", (30, 170), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

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

