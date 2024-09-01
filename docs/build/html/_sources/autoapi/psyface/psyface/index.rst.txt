psyface.psyface
===============

.. currentmodule:: source.psyface


Module Summary
^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   psyface.mask_face_region
   psyface.occlude_face_region
   psyface.extract_color_channel_means
   psyface.shift_color_temp
   psyface.face_color_shift


Functions
---------


.. py:function:: mask_face_region(input_dir: str, output_dir: str, mask_type: int = FACE_SKIN_ISOLATION, with_sub_dirs: bool = False, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5, static_image_mode: bool = False) -> None

   Applies specified mask type to video files located in input_dir, then outputs masked videos to output_dir.

   :param input_dir: A path string of the directory containing videos to process.
   :type input_dir: str
   :param output_dir: A path string of the directory where processed videos will be written to.
   :type output_dir: str
   :param mask_type: An integer indicating the type of mask to apply to the input videos. This can be one of two options:
                     either 1 for FACE_OVAL, or 2 for FACE_SKIN_ISOLATION.
   :type mask_type: int
   :param with_sub_dirs: Indicates if the input directory contains subfolders.
   :type with_sub_dirs: bool
   :param min_detection_confidence: A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe
                                    FaceMesh constructor.
   :type min_detection_confidence: float
   :param min_tracking_confidence: A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe
                                   FaceMesh constructor.
   :type min_tracking_confidence: float
   :param static_image_mode: A boolean flag indicating to the mediapipe FaceMesh that it is working with static images rather than
                             video frames.
   :type static_image_mode: bool

   :raises ValueError: Given an unknown mask type.
   :raises TypeError: Given invalid parameter types.
   :raises OSError:: Given invalid path strings for in/output directories


.. py:function:: occlude_face_region(input_dir: str, output_dir: str, landmarks_to_occlude: list[list[tuple]] | list[tuple], occlusion_fill: int = OCCLUSION_FILL_BLACK, with_sub_dirs: bool = False, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5, static_image_mode: bool = False) -> None

   For each video or image contained within the input directory, the landmark regions contained within landmarks_to_occlude
   will be occluded with either black or the facial mean pixel value. Processed files are then output to Occluded_Video_Output
   within the specified output directory.

   :param input_dir: A path string to the directory containing files to process.
   :type input_dir: str
   :param output_dir: A path string to the directory where processed videos will be written.
   :type output_dir: str
   :param landmarks_to_occlude: A list of facial landmark paths, either created by the user using utils.create_path(), or selected from the
                                predefined set of facial landmark paths.
   :type landmarks_to_occlude: list of list, list of tuple
   :param occlusion_fill: An integer flag indicating the fill method of the occluded landmark regions. One of OCCLUSION_FILL_BLACK or
                          OCCLUSION_FILL_MEAN.
   :type occlusion_fill: int
   :param with_sub_dirs: A boolean flag indicating if the input directory contains subfolders.
   :type with_sub_dirs: bool
   :param min_detection_confidence: A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe
                                    FaceMesh constructor.
   :type min_detection_confidence: float
   :param min_tracking_confidence: A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe
                                   FaceMesh constructor.
   :type min_tracking_confidence: float
   :param static_image_mode: A boolean flag indicating to the mediapipe FaceMesh that it is working with static images rather than
                             video frames.
   :type static_image_mode: bool

   :raises TypeError: Given invalid parameter types.
   :raises ValueError: Given invalid landmark sets or unrecognized fill options.
   :raises OSError: Given invalid path strings to either input_dir or output_dir.


.. py:function:: extract_color_channel_means(input_dir: str, output_dir: str, color_space: int | str = COLOR_SPACE_RGB, with_sub_dirs: bool = False, mask_face: bool = True, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> None

   Extracts and outputs mean values of each color channel from the specified color space. Creates a new directory
   'CSV_Output', where a csv file will be written for each input video file provided.

   :param input_dir: A path string to a directory containing the video files to be processed.
   :type input_dir: str
   :param output_dir: A path string to a directory where outputted csv files will be written to.
   :type output_dir: str
   :param color_space: A specifier for which color space to operate in.
   :type color_space: int, str
   :param with_sub_dirs: Indicates whether the input directory contains subfolders.
   :type with_sub_dirs: bool
   :param mask_face: Indicates whether to mask the face region prior to extracting color means.
   :type mask_face: bool
   :param min_detection_confidence: A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe
                                    FaceMesh constructor.
   :type min_detection_confidence: float
   :param min_tracking_confidence: A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe
                                   FaceMesh constructor.
   :type min_tracking_confidence: float

   :raises TypeError: Given invalid parameter types.
   :raises ValueError: Given an unrecognized color space.
   :raises OSError: If input or output directories are invalid paths.


.. py:function:: shift_color_temp(img: cv2.typing.MatLike, img_mask: cv2.typing.MatLike | None, shift_weight: float, max_color_shift: float = 8.0, max_sat_shift: float = 0.0, shift_color: str | int = COLOR_RED, sat_only: bool = False) -> cv2.typing.MatLike

   Takes in an image and a mask of the same shape, and shifts the specified color temperature by (weight*max_shift)
   units in the masked region of the image. This function makes use of the CIE Lab* perceptually uniform color space to
   perform natural looking color shifts on the face.

   :param img: An input still image or video frame.
   :type img: Matlike
   :param img_mask: A binary image with the same shape as img.
   :type img_mask: Matlike
   :param shift_weight: The current shifting weight; a float in the range [0,1] returned from a timing function.
   :type shift_weight: float
   :param max_color_shift: The maximum units to shift a* (red-green) or b* (blue-yellow) of the Lab* color space.
   :type max_color_shift: float
   :param max_sat_shift: The maximum units to shift the images saturation by.
   :type max_sat_shift: float
   :param shift_color: An integer or string literal specifying which color will be applied to the input image.
   :type shift_color: str, int
   :param sat_only: A boolean flag that indicates if only the saturation is being modified.
   :type sat_only: bool

   :raises TypeError: On invalid input parameter types.
   :raises ValueError: If an undefined color value is passed, or non-matching image and mask shapes are provided.

   :returns: **result** -- The input image, color-shifted in the region specified by the input mask.
   :rtype: Matlike


.. py:function:: face_color_shift(input_dir: str, output_dir: str, onset_t: float = 0.0, offset_t: float = 0.0, max_color_shift: float = 8.0, max_sat_shift: float = 0.0, timing_func: Callable[Ellipsis, float] = sigmoid, shift_color: str | int = COLOR_RED, with_sub_dirs: bool = False, sat_only: bool = False, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5, static_image_mode: bool = False) -> None

   For each video file contained in input_dir, the function applies a weighted color shift to the face region,
   outputting each resulting video to output_dir. Weights are calculated using a passed timing function, that returns
   a float in the normalised range [0,1].

   .. important::
      There is currently no checking or error handling performed to ensure that timing function outputs are normalised. 
      Passing in a timing function with non-normalised outputs will lead to unexpected behaviour. 

   :param input_dir: A path string to the directory containing input video files.
   :type input_dir: str
   :param output_dir: A path string to the directory where outputted video files will be saved.
   :type output_dir: str
   :param onset_t: The onset time of the colour shifting.
   :type onset_t: float
   :param offset_t: The offset time of the colour shifting.
   :type offset_t: float
   :param max_color_shift: The maximum units to shift the colour temperature by, during peak onset.
   :type max_color_shift: float
   :param max_sat_shift: The maximum units to shift the images saturation by, during peak onset.
   :type max_sat_shift: float
   :param timingFunc: Any function that takes at least one input float (time), and returns a float.
   :type timingFunc: Function() -> float
   :param shift_color: Either a string literal specifying the color of choice, or a predefined integer constant.
   :type shift_color: str, int
   :param with_sub_dirs: A boolean flag indicating whether the input directory contains nested directories.
   :type with_sub_dirs: bool
   :param sat_only: A boolean flag indicating if only the saturation of the input file will be shifted.
   :type sat_only: bool
   :param min_detection_confidence: A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe
                                    FaceMesh constructor.
   :type min_detection_confidence: float
   :param min_tracking_confidence: A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe
                                   FaceMesh constructor.
   :type min_tracking_confidence: float
   :param static_image_mode: A boolean flag indicating to the mediapipe FaceMesh that it is working with static images rather than
                             video frames.
   :type static_image_mode: bool

   :raises TypeError: Given invalid parameter types.
   :raises OSError: Given invalid directory paths.
   :raises ValueError:: If provided timing_func does not return a normalised float value.