## Overview
Currently, this project includes an application built on top of OpenCV and MediaPipe, making use of computer vision techniques to mask and isolate the facial skin region of the face from a given video.
MediaPipe's Face Mesh task provides automated detection of 468 unique facial landmarks. By accessing these landmarks for each individual video frame. The x,y coordinates of the landmarks can be accessed. 
These coordinates are used to define enclosing polygons for the facial skin regions, which are then used to create image masks to remove non-facial skin components from the resulting image. 
