import psyface.psyface as pf
from psyface.psyfaceutils import *
import cv2 as cv
import matplotlib.pyplot as plt


#in_dir = "C:\\Users\\gavin\\Desktop\\OpenCV\\Video_Song_Actors_01-24\\Video_Song_Actor_01\\Actor_01\\01-02-01-01-01-02-01.mp4"
in_dir = "C:\\Users\\gavin\\Desktop\\OpenCV\\images\\actor_05.png"
out_dir = "C:\\Users\\gavin\\Desktop\\OpenCV\\images"

#TODO potentially reprocess colour data with cv.COLOR_BGR2HSV_FULL for full range of hue values
#pf.mask_face_region(input_dir=in_dir, output_dir=out_dir, mask_type=FACE_SKIN_ISOLATION)
#pf.face_color_shift(input_dir=in_dir, output_dir=out_dir, shift_color="green", max_color_shift=8.0, max_sat_shift=-15.0, static_image_mode=True)
#pf.occlude_face_region(in_dir, out_dir, [NOSE_PATH], OCCLUSION_FILL_BLACK)
#pf.extract_color_channel_means(in_dir, out_dir)