# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- v0.3.1 ensure functions perform as expected with static images as input (bug fixes)
- v0.3.2 unit testing
- v1.0 gui preview prior to file processing

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
