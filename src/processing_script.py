import facial_manipulation as fi
import numpy as np
import os

# shift blue
#in_dir_2 = "C:\\Users\\gavin\\Desktop\\OpenCV\\Video_Song_Actors_01-24\\Video_Song_Actor_04\\Actor_04\\02-02-01-01-01-01-04.mp4"
in_dir = "C:/Users/gavin/Desktop/OpenCV/Video_Song_Actors_01-24/Video_Song_Actor_07/Actor_07/01-02-01-01-01-01-07.mp4"
out_dir = os.getcwd()

#TODO potentially reprocess colour data with cv.COLOR_BGR2HSV_FULL for full range of hue values
fi.mask_face_region(input_dir=in_dir, output_dir=out_dir)
#face_color_shift(input_dir=in_dir, output_dir=out_dir, onset_t=1.0, offset_t=3.0, max_sat_shift=-15.0, sat_only=True)
