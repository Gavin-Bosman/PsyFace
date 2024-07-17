import cv2 as cv

# opencv read images in with imread
# works with virtually any image format
img = cv.imread('Photos/portrait_1.webp')
# display the image in a new window with imshow
# provide a caption to the window with the first arg
cv.imshow('Portrait', img)

# tells the window to wait indefinitely before closing
cv.waitKey(0)

#### Resizing and rescaling images ####
# generally it is best practice to downscale your images and frames
# in order to reduce processing time

def rescaleFrame(frame, scale=0.75):
    # frame.shape[0] = height, shape[1] = width
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dim = (width,height)

    # you can alternatively provide the x,y scaling factors to the 
    # cv2.resize function itself
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    # various interpolation types; INTER_AREA is generally used to shrink an image, 
    # in order to scale an image up, use INTER_LINEAR/INTER_CUBIC

rescaled = rescaleFrame(img, scale=0.5)
cv.imshow('Scaled img', rescaled)
cv.waitKey(0)