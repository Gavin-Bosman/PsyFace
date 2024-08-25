# PsyFace
[![Documentation Status](https://readthedocs.org/projects/gavin-bosman-psyface/badge/?version=latest)](https://gavin-bosman-psyface.readthedocs.io/en/latest/?badge=latest)

A python tool kit for performing facial psychology experiments and analysis. This project is designed to allow psychology researchers to perform previously complex facial manipulation tasks in as little as 2 lines of code. 

## Underlying Model
MediaPipe's Face Mesh task provides automated detection of 468 unique facial landmarks. By accessing the x-y screen coordinates of these landmarks, many complex image and video operations can be performed. 

[![MediaPipe FaceMesh](https://ai.google.dev/static/mediapipe/images/solutions/examples/face_landmark.png)](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)

For more on mediapipe, see [here](https://ai.google.dev/edge/mediapipe/solutions/guide)

## A Quick Example: Facial Occlusion
```
import PsyFace as pf

# For windows operating systems use "\\" to delimit your pathstrings
# For linux or macos use "/" to delimit your pathstrings
in_dir = ".\<your input directory>"
out_dir = ".\<your output directory>"

# Psyface provides an extensive list of predefined landmark paths, as well as several predefined occlusion types
pf.occlude_face_region(input_dir = in_dir, output_dir = out_dir, landmarks_to_occlude = [LEFT_EYE_PATH, RIGHT_EYE_PATH], occlusion_fill = OCCLUSION_FILL_BAR)
```

<img src="images/portrait.jpg", alt="Unprocessed Image", width="150">
<img src="images/portrait_occluded.jpg", alt="Occluded Image", width="150">

## Documentation and Changelog

This project maintains a changelog, following the format of [keep a changelog](https://keepachangelog.com/en/1.0.0/). This project also adheres to [semantic versioning](https://semver.org/spec/v2.0.0.html).

To view our documentation, examples and tutorials, see [PsyFace Docs](https://gavin-bosman-psyface.readthedocs.io/en/latest/).

## Contributing

PsyFace is a young project, and it is still far from perfect! If you have any ideas on how to improve the package please submit an inquiry and we will work on implementing it right away!

Pull requests are always welcome. If you spot a mistake, error, or inefficiency within the source code, let us know!

## license

[MIT](https://opensource.org/license/mit)
