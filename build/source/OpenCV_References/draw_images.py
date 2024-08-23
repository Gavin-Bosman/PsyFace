import cv2 as cv
import numpy as np

# create a blank image 500 height x 500 width x 3 color channels
blank = np.zeros((500,500,3), dtype='uint8')

# 1. paint background colour
# set all pixels to green
# blank[:] = 0,255,0
# cv.imshow('Green', blank)

# 2. generate a coloured region
# blank[200:300, 200:250] = 125,0,125

# 3. drawing shapes using cv functions
# thickness param can be set to an integer for line weight, or cv.FILLED
cv.rectangle(blank, (100,100), (200,200), (125,0,125), thickness=cv.FILLED)

cv.circle(blank, (300,400), 40, (0, 125, 175), thickness=-1)
cv.imshow('Shapes', blank)

# see also cv.line(matlike, start, end, colour, thickness)

cv.waitKey(0)