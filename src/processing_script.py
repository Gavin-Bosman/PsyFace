import facial_manipulation as fi
from utils import *
import numpy as np
import os
import cv2

in_dir = "C:\\Users\\gavin\\Desktop\\OpenCV\\Video_Song_Actors_01-24\\Video_Song_Actor_01\\Actor_01\\01-02-01-01-01-02-01.mp4"
out_dir = os.getcwd()

#TODO potentially reprocess colour data with cv.COLOR_BGR2HSV_FULL for full range of hue values
#fi.mask_face_region(input_dir=in_dir, output_dir=out_dir)
#fi.face_color_shift(input_dir=in_dir, output_dir=out_dir, onset_t=1.0, offset_t=6.0, shift_color="red", max_color_shift=20.0)
fi.occlude_face_region(in_dir, out_dir, [NOSE_PATH], OCCLUSION_FILL_BAR)
