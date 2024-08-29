import psyface.psyface as pf
from psyface.psyfaceutils import *
import os


#in_dir = "C:\\Users\\gavin\\Desktop\\OpenCV\\Video_Song_Actors_01-24\\Video_Song_Actor_01\\Actor_01\\01-02-01-01-01-02-01.mp4"
in_dir = "C:\\Users\\gavin\\Desktop\\OpenCV\\images\\portrait.jpg"
out_dir = "C:\\Users\\gavin\\Desktop\\OpenCV\\images"

#TODO potentially reprocess colour data with cv.COLOR_BGR2HSV_FULL for full range of hue values
#pf.mask_face_region(input_dir=in_dir, output_dir=out_dir, mask_type=FACE_OVAL)
#pf.face_color_shift(input_dir=in_dir, output_dir=out_dir, onset_t=0.0, offset_t=3.0, shift_color="red", max_color_shift=20.0)
pf.occlude_face_region(in_dir, out_dir, [LEFT_EYE_PATH], OCCLUSION_FILL_BAR)
#pf.extract_color_channel_means(in_dir, out_dir)
