psyface.psyfaceutils
====================

.. py:module:: psyface.psyfaceutils


Module Summary
^^^^^^^^^^^^^^


.. autoapisummary::

   psyface.psyfaceutils.create_path
   psyface.psyfaceutils.get_min_max_bgr
   psyface.psyfaceutils.transcode_video_to_mp4


Functions
---------


.. py:function:: create_path(landmark_set: list[int]) -> list[tuple]

   Given a list of facial landmarks (int), returns a list of tuples, creating a closed path in the form
   [(a,b), (b,c), (c,d), ...]. This function allows the user to create custom facial landmark sets, for use in
   mask_face_region() and occlude_face_region().

   :param landmark_set: A python list containing facial landmark indicies.
   :type landmark_set: list of int

   :returns: **routes** -- A list of tuples containing overlapping points, forming a path.
   :rtype: list of tuple


.. py:function:: transcode_video_to_mp4(input_dir: str, output_dir: str, with_sub_dirs: bool = False) -> None

   Given an input directory containing one or more video files, transcodes all video files from their current
   container to mp4. This function can be used to preprocess older video file types before masking, occluding or color shifting.

   :param input_dir: A path string to the directory containing the videos to be transcoded.
   :type input_dir: str
   :param output_dir: A path string to the directory where transcoded videos will be written too.
   :type output_dir: str
   :param with_sub_dirs: A boolean flag indicating if the input directory contains sub-directories.
   :type with_sub_dirs: bool

   :raises TypeError: Given invalid parameter types.
   :raises OSError: Given invalid paths for input_dir or output_dir.


.. py:function:: get_min_max_bgr(filePath: str, focusColor: int | str = COLOR_RED) -> tuple

   Given an input video file path, returns the minimum and maximum (B,G,R) colors, containing the minimum and maximum
   values of the focus color.

   :param filePath: The path string of the location of the file to be processed.
   :type filePath: str
   :param focusColor: The RGB color channel to focus on. Either one of the predefined color constants, or a string literal of the colors name.
   :type focusColor: int, str

   :raises TypeError: Given invalid parameter types.
   :raises ValueError: Given a nonexisting file path, or a non RGB focus color.

   :returns: * **min_color** (*array of int*) -- A BGR colour code (ie. (100, 105, 80)) containing the minimum value of the focus color.
             * **max_color** (*array of int*) -- A BGR colour code (ie. (100, 105, 80)) containing the minimum value of the focus color.


Predefined Constants and Formatting Options
-------------------------------------------


Facial Landmark Sets
^^^^^^^^^^^^^^^^^^^^


.. note::
   The following landmark indice sets are for use with the create_path() function. Predefined landmark paths are also 
   available, but these indice sets provide quick and easy reference for the MediaPipe Facial Landmarker.

.. py:data:: LEFT_EYE_IDX
   :value: [301, 334, 296, 336, 285, 413, 464, 453, 452, 451, 450, 449, 448, 261, 265, 383, 301]
.. py:data:: LEFT_IRIS_IDX
   :value: [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263]
.. py:data:: RIGHT_EYE_IDX
   :value: [71, 105, 66, 107, 55, 189, 244, 233, 232, 231, 230, 229, 228, 31, 35, 156, 71]
.. py:data:: RIGHT_IRIS_IDX
   :value: [33, 7, 163, 144, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
.. py:data:: NOSE_IDX
   :value: [168, 193, 122, 196, 3, 198, 49, 203, 167, 164, 393, 423, 279, 420, 248, 419, 351, 417, 168]
.. py:data:: LIPS_IDX
   :value: [164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57, 186, 92, 165, 167, 164]
.. py:data:: LIPS_TIGHT_IDX
   :value: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 0, 37, 39, 40, 185, 61]
.. py:data:: FACE_OVAL_IDX
   :value: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,...
.. py:data:: FACE_OVAL_TIGHT_IDX
   :value: [10, 338, 297, 332, 284, 251, 389, 356, 345, 352, 376, 433, 397, 365, 379, 378, 400, 377, 152,...

.. important:: 
   PsyFace's predefined landmark paths use the same naming convention as the indice lists, but are appended with 
   '_PATH' rather than '_IDX'. For every landmark indice set there is a landmark path of the same prefix.


Masking Options
^^^^^^^^^^^^^^^

.. py:data:: FACE_OVAL
.. py:data:: FACE_OVAL_TIGHT
.. py:data:: FACE_SKIN_ISOLATION
.. py:data:: MASK_OPTIONS
   :value: ['FACE_OVAL', 'FACE_OVAL_TIGHT', 'FACE_SKIN_ISOLATION']


Color Options
^^^^^^^^^^^^^

.. py:data:: COLOR_SPACE_RGB
.. py:data:: COLOR_SPACE_HSV
.. py:data:: COLOR_SPACE_GRAYSCALE
.. py:data:: COLOR_SPACES
   :value: ['COLOR_SPACE_RGB', 'COLOR_SPACE_HSV', 'COLOR_SPACE_GRAYSCALE']
.. py:data:: COLOR_RED
.. py:data:: COLOR_BLUE
.. py:data:: COLOR_GREEN
.. py:data:: COLOR_YELLOW


Occlusion Fill Options
^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: OCCLUSION_FILL_BLACK
.. py:data:: OCCLUSION_FILL_MEAN
.. py:data:: OCCLUSION_FILL_BAR