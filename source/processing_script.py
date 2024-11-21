import pyfame.pyfame as pf
from pyfame.pyfameutils import *
import cv2 as cv
import matplotlib.pyplot as plt

#in_dir = "C:\\Users\\gavin\\Desktop\\OpenCV\\Video_Song_Actors_01-24\\Video_Song_Actor_08\\Actor_08\\01-02-05-01-01-01-08.mp4"
in_dir = "C:\\Users\\gavin\\Desktop\\OpenCV\\images\\Actor_08.png"
out_dir = "C:\\Users\\gavin\\Desktop\\OpenCV\\images"

#TODO potentially reprocess colour data with cv.COLOR_BGR2HSV_FULL for full range of hue values
#pf.mask_face_region(input_dir=in_dir, output_dir=out_dir, mask_type=HEMI_FACE_BOTTOM_MASK, background_color=(125,125,255))
#pf.face_color_shift(input_dir=in_dir, output_dir=out_dir, shift_color="red", landmark_regions=[LEFT_CHEEK_PATH, CHIN_PATH], shift_magnitude=15.0)
#pf.occlude_face_region(in_dir, out_dir, [FACE_SKIN_PATH], OCCLUSION_FILL_BLACK)
#pf.extract_face_color_means(in_dir, out_dir)
#pf.face_brightness_shift(input_dir=in_dir, output_dir=out_dir, shift_magnitude=-30, landmark_regions=[HEMI_FACE_TOP_PATH])
#pf.face_saturation_shift(input_dir=in_dir, output_dir=out_dir, shift_magnitude=-20.0, landmark_regions=[LEFT_CHEEK_PATH, CHIN_PATH])
#pf.blur_face_region(input_dir=in_dir, output_dir=out_dir, blur_method="Gaussian", k_size=91)
pf.apply_noise(input_dir=in_dir, output_dir=out_dir, noise_method=NOISE_METHOD_SALT_AND_PEPPER, rand_seed=1234)

# Creating pyplot style grid of outputs

fig = plt.figure(figsize=(5,5))

im1 = cv.cvtColor(cv.imread("images\\Actor_08.png"), cv.COLOR_BGR2RGB)
im1 = cv.copyMakeBorder(im1, top=5, bottom=5, left=5, right=5, borderType=cv.BORDER_CONSTANT, value=0)
im2 = cv.cvtColor(cv.imread("images\\Actor_08_color_shifted.png"), cv.COLOR_BGR2RGB)
im2 = cv.copyMakeBorder(im2, top=5, bottom=5, left=5, right=5, borderType=cv.BORDER_CONSTANT, value=0)
im3 = cv.cvtColor(cv.imread("images\\Actor_08_sat_brightened.png"), cv.COLOR_BGR2RGB)
im3 = cv.copyMakeBorder(im3, top=5, bottom=5, left=5, right=5, borderType=cv.BORDER_CONSTANT, value=0)

im4 = cv.cvtColor(cv.imread("images\\Actor_01_occluded_bar.png"), cv.COLOR_BGR2RGB)
im4 = cv.copyMakeBorder(im4, top=5, bottom=5, left=5, right=5, borderType=cv.BORDER_CONSTANT, value=0)
im5 = cv.cvtColor(cv.imread("images\\Actor_01_occluded_unilateral.png"), cv.COLOR_BGR2RGB)
im5 = cv.copyMakeBorder(im5, top=5, bottom=5, left=5, right=5, borderType=cv.BORDER_CONSTANT, value=0)
im6 = cv.cvtColor(cv.imread("images\\Actor_01_blurred_gaussian.png"), cv.COLOR_BGR2RGB)
im6 = cv.copyMakeBorder(im6, top=5, bottom=5, left=5, right=5, borderType=cv.BORDER_CONSTANT, value=0)

im7 = cv.cvtColor(cv.imread("images\\Actor_04_masked_oval.png"), cv.COLOR_BGR2RGB)
im7 = cv.copyMakeBorder(im7, top=5, bottom=5, left=5, right=5, borderType=cv.BORDER_CONSTANT, value=0)
im8 = cv.cvtColor(cv.imread("images\\Actor_04_masked_skin.png"), cv.COLOR_BGR2RGB)
im8 = cv.copyMakeBorder(im8, top=5, bottom=5, left=5, right=5, borderType=cv.BORDER_CONSTANT, value=0)
im9 = cv.cvtColor(cv.imread("images\\Actor_04_masked_features.png"), cv.COLOR_BGR2RGB)
im9 = cv.copyMakeBorder(im9, top=5, bottom=5, left=5, right=5, borderType=cv.BORDER_CONSTANT, value=0)

fig.add_subplot(3, 3, 1)
plt.imshow(im1)
plt.axis('off')
plt.title('Original', fontsize=10)

fig.add_subplot(3,3,2)
plt.imshow(im2)
plt.axis('off')
plt.title('Color-shifted', fontsize=10)

fig.add_subplot(3,3,3)
plt.imshow(im3)
plt.axis('off')
plt.title('Desaturated & \nBrightness shifted', fontsize=10)

fig.add_subplot(3,3,4)
plt.imshow(im4)
plt.axis('off')
plt.title('Landmark Occlusion', fontsize=10)

fig.add_subplot(3,3,5)
plt.imshow(im5)
plt.axis('off')
plt.title('Unilateral Occlusion', fontsize=10)

fig.add_subplot(3,3,6)
plt.imshow(im6)
plt.axis('off')
plt.title('Facial blur - Gaussian', fontsize=10)

fig.add_subplot(3,3,7)
plt.imshow(im7)
plt.axis('off')
plt.title('Face masking', fontsize=10)

fig.add_subplot(3,3,8)
plt.imshow(im8)
plt.axis('off')
plt.title('Skin masking', fontsize=10)

fig.add_subplot(3,3,9)
plt.imshow(im9)
plt.axis('off')
plt.title('Feature masking', fontsize=10)

plt.show()