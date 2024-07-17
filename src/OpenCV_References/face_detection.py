import cv2 as cv
import mediapipe as mp

capture = cv.VideoCapture('Videos/project.mp4')

# mediaPipe face detection modules
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
# The FaceDetection class can take a confidence level as a parameter
# the default confidence is 0.5
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, frame = capture.read()
    height, width, *_ = frame.shape
    dims = (int(width//1.5), int(height//1.5))
    frame = cv.resize(frame, dims, interpolation=cv.INTER_AREA)

    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        # detection class provides bounding box information of the face
        # as well as landmark locations
        for id, detection, in enumerate(results.detections):
            mpDraw.draw_detection(frame, detection)
            
            # bounding box values are normalised, need to convert to pixel values
            bbox_normalised = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            bbox = int(bbox_normalised.xmin * iw), int(bbox_normalised.ymin * ih), \
                int(bbox_normalised.width * iw), int(bbox_normalised.height * ih)

            # use the denormalised bounding box coords to modify or create your own bounding box
            print(bbox)

    cv.imshow('Video', frame)
    
    if cv.waitKey(10) == ord('x'):
        cv.destroyAllWindows()
        break

capture.release()