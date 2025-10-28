# Very basic ArUco tag reader
import cv2 as cv

capture = cv.VideoCapture(0)
#uncomment if u want to change the window size:
#capture.set(cv.CAP_PROP_FRAME_WIDTH,1280)

aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
parameters = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

if not capture.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = capture.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #grayscale helps with detection?
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(frame)

    output_image = frame.copy()
    cv.aruco.drawDetectedMarkers(output_image,corners,ids)
    cv.aruco.drawDetectedMarkers(output_image, rejected, borderColor=(100,0,240))
    cv.imshow("markers",output_image)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
capture.release()
cv.destroyAllWindows()