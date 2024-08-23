import cv2 as cv
import numpy as np

img = cv.imread('Photos/Portrait_1.webp')
cv.imshow('Portrait', img)

# implementing translation
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dims = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dims)

# implementing rotation
def rotate(img, angle, rotPoint=None):
    (height,width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dims = (width,height)

    return cv.warpAffine(img, rotMat, dims)