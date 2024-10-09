# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- v0.5... unit testing
- v1.0 gui preview prior to file processing

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
