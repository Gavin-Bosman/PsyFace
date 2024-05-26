import cv2 as cv

# reading in videos with videocapture
capture = cv.VideoCapture('Videos/project.mp4')

# videos are read in by opencv frame by frame
while True:
    # read returns a bool indicating if the next frame exists
    # as well as the frame itself, in a tuple
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

# scaling videos can be done with cv2.resize frame by frame
# however VideoCapture provides a set method for live video
def changeRes(capture, width, height):
    # works only with live video
    capture.set(3,width)
    capture.set(4,height)