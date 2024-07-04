###TODO vary colour shift strength according to some timing function
import facial_isolation_module as fi
import numpy as np
import os

in_dir_2 = "C:\\Users\\gavin\\Desktop\\OpenCV\\Video_Song_Actors_01-24\\Video_Song_Actor_01\\Actor_01\\02-02-04-02-01-02-01.mp4"
out_dir = os.getcwd()

fi.face_color_filter(in_dir_2, out_dir, "red")