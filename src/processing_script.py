import src.facial_manipulation as fi
import numpy as np
import os

# shift blue
#in_dir_2 = "C:\\Users\\gavin\\Desktop\\OpenCV\\Video_Song_Actors_01-24\\Video_Song_Actor_04\\Actor_04\\02-02-01-01-01-01-04.mp4"
in_dir_2 = "C:\\Users\\gavin\\Desktop\\OpenCV\\Video_Song_Actors_01-24\\Video_Song_Actor_01\\Actor_01\\02-02-05-02-01-02-01.mp4"
out_dir = os.getcwd()

#TODO potentially reprocess colour data with cv.COLOR_BGR2HSV_FULL for full range of hue values

fi.face_color_shift(input_dir=in_dir_2, output_dir=out_dir, onset_t=1.0, offset_t=3.5, max_color_shift= 10.0, shift_color="red")
