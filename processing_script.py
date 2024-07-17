import facial_isolation_module as fi
import numpy as np
import os

# shift blue
#in_dir_2 = "C:\\Users\\gavin\\Desktop\\OpenCV\\Video_Song_Actors_01-24\\Video_Song_Actor_04\\Actor_04\\02-02-01-01-01-01-04.mp4"
in_dir_2 = "C:\\Users\\gavin\\Desktop\\OpenCV\\Video_Song_Actors_01-24\\Video_Song_Actor_01\\Actor_01\\02-02-05-02-01-02-01.mp4"
out_dir = os.getcwd()

#TODO potentially reprocess colour data with cv.COLOR_BGR2HSV_FULL for full range of hue values

fi.face_color_shift(inputDirectory=in_dir_2, outputDirectory=out_dir, onset=1.0, offset=3.5, maxShift= 10.0, filterColor="red")
