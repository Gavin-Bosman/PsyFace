# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- v0.3 occlude_face_region
- v0.3 test cases

## [0.2.0] - 2024-07-21

### Added

- v0.2 added dependancy ffprobe-python

### Changed

- Added input file codec sniffing, output video files will now match input type for mask_face_region
and face_color_shift

### Removed

## [0.1.1] - 2024-07-20

### Added

### Changed

- Minor bug fix for negative saturation shift

### Removed

## [0.1.0] - 2024-07-17

### Added

- v0.1 mask_face_region
- v0.1 extract_color_channel_means
- v0.1 face_color_shift
- v0.1 shift_color_temp

### Changed

- Updated documentation and type hints for all package functions.
- Vectorized color shifting operations in shift_color_temp, massively reducing time costs.
- Restructured package into src, data and testing folders.
- Moved constants and helper functions into utils.py

### Removed
