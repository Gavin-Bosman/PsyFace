# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- v0.5.8-0.5.9 inversion and scrambling of facial landmarks, bubble occlusion option for `occlude_face_region`
- v0.6... unit testing
- v1.0 gui preview prior to file processing

## [0.5.7] - 2024-11-20

### Added

- New function `apply_noise()` has been implemented. This function provides three noise operations to select from; 
'pixelate', 'gaussian' and 'salt and pepper'. 
- `apply_noise()` provides a variety of customization options such as specifying the noise probability, mean and standard
deviation of the gaussian curve to be sampled from, as well as a random seed to be passed to the numpy random number generator. 
- An expanded set of masking options has been added to pyfameutils.MASK_OPTIONS for use with `mask_face_region()`. `mask_face_region` now also allows the user to specify the background color of output files via a BGR integer color code. 

### Changed

- Previous masking options FACE_OVAL and FACE_OVAL_TIGHT have been removed and replaced by the singular FACE_OVAL_MASK in hopes to alleviate any user confusion between the two options previously. 
- PsyFace has been officially renamed to PyFAME: the Python Facial Analysis and Manipulation Environment.

### Removed

## [0.5.6] - 2024-11-12

### Added

- Bug fixes for last major feature update (v0.5.5).
- `CHIN_PATH` has been added as a predefined path for use with all facial manipulation functions.
- `extract_color_channel_means` has been renamed as `extract_face_color_means`. The function now will not only output full-facial means, but also regional color means in the cheek, nose and chin areas for all colour spaces. 

### Changed

- The Hemi-face family of landmark paths have been converted to standard paths, and no longer require in-place computation. 
- `occlude_face_region`'s implementation of bar-style occlusion has been reworked, such that now the occluding bar will track correctly with the position of the face and axis of the head (the occluding bar no longer remains paralell to the horizontal axis).
- `face_color_shift`, `face_saturation_shift` and `face_brightness_shift` now only take list[list[tuple]] for input parameter `landmark_regions`. This massively reduces the ammount of duplicate code previously divided among if-else statements based on what was passed to `landmark_regions`.

### Removed

## [0.5.3 - 0.5.5] - 2024-10-16

### Added

- Major feature updates to all facial manipulation functions. `face_color_shift`, `face_saturation_shift`, `face_brightness_shift` and `blur_face_region` now are all compatible with timing functions, and every predefined landmark region defined within `psyfaceutils.py`. 
- Some of the landmark paths have been redefined as placeholders, as they either need to be calculated in place (hemi-face regions) or require a different method to draw the landmark polygons (Cheek landmark regions form concave polygons).
- `FACE_SKIN_PATH` constant has been defined in order to provided easier access to facial skin colouring, leaving the lips and eyes untouched. For similar ease of use reasons, other commonly used grouped regions have been defined, including `CHEEKS_PATH` and `CHEEKS_NOSE_PATH` for use in facial "blushing".

### Changed

### Removed

- Bad practice global variable declarations have been removed entirely. 

## [0.5.2] - 2024-10-09

### Added

- `Occlude_face_region` can now perform vertical and horizontal hemi-face occlusion. Hemi-face masks rely on the facial screen coords, thus they cannot be precomputed. However, predefined placeholder constants `HEMI_FACE_TOP`, `HEMI_FACE_BOTTOM`, `HEMI_FACE_LEFT`, and `HEMI_FACE_RIGHT` have been defined and can still be passed in `landmarks_to_occlude` as any of the other predefined landmark paths can. 
- Helper function `compute_line_intersection` has been created and can be found in `psyfaceutils.py`.
- Additional masking option `EYES_NOSE_MOUTH_MASK` has been added to `mask_face_region`.

### Changed

### Removed

- Predefined landmark paths `UPPER_FACE_PATH` and `LOWER_FACE_PATH` have been removed, and replaced with `HEMI_FACE_TOP`, and `HEMI_FACE_BOTTOM` respectively.

## [0.5.1] - 2024-10-03

### Added

- `Blur_face_region` provides dynamic facial blurring functionality with several blurring methods (average, gaussian, median) over user-specified facial regions. 
- Added horizontal hemi-face occlusion, `UPPER_FACE_PATH` and `LOWER_FACE_PATH` constants can be found in psyfaceutils.py.

### Changed

- `Face_luminance_shift` has been replaced with `face_brightness_shift`. `Face_brightness_shift` will now take an integer shift value in the range [-255, 255], with -255 and 255 representing pure black and white respectively. 

### Removed

- `Face_luminance_shift` has been removed due to buggy behaviour when manipulating image luminance.

## [0.5.0] - 2024-09-24

### Added

- Package documentation is now built with MKDocs
- `Face_saturation_shift` and `Face_luminance_shift` are now standalone functions, where previously saturation and luma parameters were passed to the Face_color_shift function. 
- Github.io hosting for documentation page, as well as refactored github landing page and readme.md.
- License.txt added to root project structure.

### Changed

- `Shift_color_temp` was refactored to be a nested function within `Face_color_shift`, saturation and luminance shifting were relocated to their own specific functions. 
- Floodfilling operation involved with foreground-background seperation had some buggy behaviour if there was any discontinuity in the background. An intermediate step was added where prior to floodfilling, the thresholded image is padded with a 10 pixel border, which is removed after the floodfill. This border ensures background continuity when performing the floodfill operation.
- Parameters `max_color_shift` and `max_sat_shift` are now renamed to `shift_magnitude`.

### Removed

- Sphinx and readthedocs project files and dependencies

## [0.4.2] - 2024-08-24

### Added

- Sphinx dependency for autodocumentation.
- Rst files defining the documentation build. 

### Changed

- Updated readme.md with examples, licenses and link to documentation page

### Removed

## [0.4.1] - 2024-08-18

### Added

### Changed

- v0.4.1 bug fixes for processing directories of mixed file types (images and videos). 

### Removed

## [0.4.0] - 2024-08-17

### Added

### Changed

- v0.4 Refactored all methods; moved repetative frame operations to nested functions for increased readability.
- Fixed buggy behaviour when working with still images over all methods. On top of video formats .MP4 and .MOV, you can now perform facial masking, occlusion and colour shifting over image formats .jpg, .jpeg, .png, and .bmp.
- Increased error handling; methods should now be able to process large directories of mixed file formats efficiently in a single call. 

### Removed

## [0.3.1] - 2024-08-11

### Added

- v0.3.1 Support for nose masking and occluding

### Changed

- Added bar-style occlusion options to occlude_face_region(). You can now perform bar-style occlusion on the eyes, nose 
and mouth regions. 

### Removed

## [0.3.0] - 2024-08-02

### Added

- v0.3 occlude_face_region()

### Changed

- Redefined the naming convention used for constants in utils.py

### Removed

## [0.2.2] - 2024-07-31

### Added

### Changed

- Changed mp4 video codec from h264 to cv2 supported mp4v.
- Mask_face_region and face_color_shift now take confidence parameters for the underlying mediapipe face landmarker model.
- Implemented otsu thresholding to isolate foreground to use as a mask. This foreground mask ensures that no background 
artifacts are present in the facial color shifting, or facial masking. 
- Added documentation for new function parameters.

### Removed

## [0.2.1] - 2024-07-24

### Added

- v0.2.1 transcode_video_to_mp4()

### Changed

- All functions will work by default with .mp4 and .mov video files. If an older container is being used, 
see transcode_video_to_mp4 to convert video codecs.
- Bug fixes with facial mask in face_color_shift; removed background artifacts present in the masked facial region.

### Removed
- Removed dependancy ffprobe-python.

## [0.2.0] - 2024-07-21

### Added

- v0.2 added dependancy ffprobe-python.

### Changed

- Added input file codec sniffing, output video files will now match input type for mask_face_region
and face_color_shift.

### Removed

## [0.1.1] - 2024-07-20

### Added

### Changed

- Minor bug fix for negative saturation shift.

### Removed

## [0.1.0] - 2024-07-17

### Added

- v0.1 mask_face_region()
- v0.1 extract_color_channel_means()
- v0.1 face_color_shift()
- v0.1 shift_color_temp()

### Changed

- Updated documentation and type hints for all package functions.
- Vectorized color shifting operations in shift_color_temp, massively reducing time costs.
- Restructured package into src, data and testing folders.
- Moved constants and helper functions into utils.py.

### Removed
