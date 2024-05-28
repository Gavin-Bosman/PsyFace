## Overview
The goal of this project is to find statistical correlations between emotional valence and facial skin tone. 

Currently, this project includes an application built on top of OpenCV and MediaPipe, making use of computer vision techniques to mask and isolate the facial skin region of the face from a given video.

MediaPipe's Face Mesh task provides automated detection of 468 unique facial landmarks. By accessing these landmarks for each individual video frame. The x,y coordinates of the landmarks can be accessed. These coordinates are used to define enclosing polygons for the facial skin regions, which are then used to create image masks to remove non-facial skin components from the resulting image. 

## Input Data
The Ryerson Audio Visual Dataset of Emotional Speech and Song (RAVDESS) contains 7356 files. The dataset contains 24 actors, each vocalizing two lexically matched statements in neutral North American English. Furthermore, each vocalization is stored in three different modalities: audio-only (AO), video-only (VO) and audio-video (AV). This project focuses on the video aspect of the RAVDESS dataset. 

## Example
The input videos used are taken from the RAVDESS dataset, an example of which can be found here:
https://github.com/Gavin-Bosman/face_skin_isolation/assets/124214234/1daae4c5-6db5-4a45-9399-f96c9ff6af86

An example of the same video post-processing can be found here:
https://github.com/Gavin-Bosman/face_skin_isolation/assets/124214234/504443e4-e29d-44e2-94fd-dfc8242e5e92
