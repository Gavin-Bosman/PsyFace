import facial_isolation_module as fi
import os

in_dir = "C:\\Users\\gavin\\Desktop\\OpenCV\\Video_Song_Actors_01-24"
out_dir = os.getcwd()

fi.mask_face_region(in_dir, out_dir, fi.FACE_SKIN_ISOLATION, True, True, fi.COLOR_SPACE_RGB)