#Script to read aruco tags and their distance. Input marker size in line 16 (meters).
import cv2 as cv
import numpy as np

def main(args=None):
    #Load Calibration Data
    try:
        calib_data = np.load("calibration_data.npz")
        mtx = calib_data['mtx']
        dist = calib_data['dist']
    except FileNotFoundError:
        print("Error: calibration_data.npz not found. Run calibration script first.")
        return

    #Define Marker Size 
    MARKER_SIZE = 0.01905  # in meters

    capture = cv.VideoCapture(0)

    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

    if not capture.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        corners, ids, rejected = detector.detectMarkers(frame)
        output_image = frame.copy()

        if ids is not None:
            cv.aruco.drawDetectedMarkers(output_image, corners, ids)

            marker_points = np.array([[-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
                                      [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
                                      [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
                                      [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]], dtype=np.float32)

            for i in range(len(ids)):
                _, rvec, tvec = cv.solvePnP(marker_points, corners[i], mtx, dist, False, cv.SOLVEPNP_IPPE_SQUARE)

                cv.drawFrameAxes(output_image, mtx, dist, rvec, tvec, 0.03)

                distance = tvec[2][0]
                dist_inches = distance / 0.0254

                text_coord = tuple(corners[i][0][0].astype(int))
                cv.putText(output_image, f"Dist: {distance:.2f}m ({dist_inches:.1f}in)", 
                           text_coord, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                print(f"ID: {ids[i]} | Distance: {distance:.3f}m")

        cv.imshow("markers", output_image)

        if cv.waitKey(1) == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
