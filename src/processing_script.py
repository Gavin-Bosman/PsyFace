import facial_manipulation as fi
from utils import transcode_video_to_mp4
import numpy as np
import os
import cv2

# shift blue
in_dir = "C:/Users/gavin/Desktop/OpenCV/Video_Song_Actors_01-24/Video_Song_Actor_07/Actor_07/01-02-01-01-01-01-07.mp4"
#in_dir = "C:/Users/gavin/Desktop/OpenCV/3762907-uhd_3840_2160_25fps.avi"
out_dir = os.getcwd()

#print(cv2.getBuildInformation())

#TODO potentially reprocess colour data with cv.COLOR_BGR2HSV_FULL for full range of hue values
fi.face_color_shift(input_dir=in_dir, output_dir=out_dir, onset_t=0.0, offset_t=3.5, max_sat_shift=-15.0, shift_color="green", max_color_shift=10.0)
