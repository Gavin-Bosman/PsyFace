import cv2 as cv

img = cv.imread('Photos/Portrait_1.webp')

# converting an image to greyscale
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# noise/blurring
blur = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)

# Edge Cascade
# cv2.canny takes a minimum and maximum threshold value to test for edges
# if a local maximum is above the max threshold, it is sure to be an edge
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny edges', canny)

# Dilating the image
dilated = cv.dilate(canny, (3,3), iterations=3)
cv.imshow('Dilated img', dilated)

# Eroding an image
eroded = cv.erode(dilated, (3,3), iterations=3)
cv.imshow('Eroded img', eroded)

# Cropping an image can be performed with basic array slicing

cv.waitKey(0)