## Overview
The goal of this project is to find statistical correlations between emotional valence and facial skin tone. 

Currently, this project includes an application built on top of OpenCV and MediaPipe, making use of computer vision techniques to mask and isolate the facial skin region of the face from a given video.

MediaPipe's Face Mesh task provides automated detection of 468 unique facial landmarks. By accessing these landmarks for each individual video frame. The x,y coordinates of the landmarks can be accessed. These coordinates are used to define enclosing polygons for the facial skin regions, which are then used to create image masks to remove non-facial skin components from the resulting image. 

## Input Data
The Ryerson Audio Visual Dataset of Emotional Speech and Song (RAVDESS) contains 7356 files. The dataset contains 24 actors, each vocalizing two lexically matched statements in neutral North American English. Furthermore, each vocalization is stored in three different modalities: audio-only (AO), video-only (VO) and audio-video (AV). This project focuses on the video aspect of the RAVDESS dataset. 

## Example
To get a visual understanding of what the face_skin_video.py program does, we feed in a video of an actor, that would look like the photo below.

![image](https://github.com/Gavin-Bosman/face_skin_isolation/assets/124214234/8d9c3d3b-ec6b-485c-8565-32da0caca509)


Post processing, each frame of the video would look like the photo below. This represents the area of the original video that is masked prior to taking the mean pixel values of the image.

![image](https://github.com/Gavin-Bosman/face_skin_isolation/assets/124214234/9e340468-e5d6-4543-8d61-5b06d120249e)

Using this facial masking, the mean pixel values, alongside the current timestamp of the video, are written to a csv file with the same name as the originally input video. 
