# API Reference

## Facial Masking

`mask_face_region` will apply the specified `mask_type` to all files contained in `input_dir`, outputting masked images and videos to `output_dir`\\Masked_Video_Output.

``` py
def mask_face_region(input_dir:str, output_dir:str, mask_type:int = FACE_SKIN_ISOLATION, with_sub_dirs:bool = False,
                min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5, static_image_mode:bool = False) -> None:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir`                | `#!python str` | A path string to the directory containing files to process. |
| `output_dir`               | `#!python str` | A path string to the directory where processed files will be output. |
| `mask_type`                | `#!python int` | An integer flag specifying the type of masking operation being performed. One of `FACE_OVAL`, `FACE_OVAL_TIGHT` OR `FACE_SKIN_ISOLATION`. |
| `with_sub_dirs`            | `#!python bool` | A boolean flag indicating if the input directory contains sub-directories. |
| `min_detection_confidence` | `#!python float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |
| `min_tracking_confidence`  | `#!python float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |
| `static_image_mode`        | `#!python bool` | A boolean flag indicating that the current filetype is static images. |

???+ info
    Parameters `min_detection_confidence`, `min_tracking_confidence`, and `static_image_mode` are passed directly to the declaration of the MediaPipe FaceMesh model. 

    ``` py
    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, min_detection_confidence = min_detection_confidence,
                                        min_tracking_confidence = min_tracking_confidence, static_image_mode = static_image_mode)
    ```

    For more information on MediaPipes FaceMesh solution, [see here.](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)

## Dynamic Facial Occlusion

`occlude_face_region` takes the landmark regions specified within `landmarks_to_occlude`, and occludes them with the specified method for each image or video file present within the input directory provided in `input_dir`. Processed videos will be written to `output_dir`\\Occluded_Video_Output.

``` py
def occlude_face_region(input_dir:str, output_dir:str, landmarks_to_occlude:list[list[tuple]] | list[tuple], 
    occlusion_fill:int = OCCLUSION_FILL_BLACK, with_sub_dirs:bool =  False, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5, static_image_mode:bool = False) -> None:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir`                | `#!python str` | A path string to the directory containing files to process. |
| `output_dir`               | `#!python str` | A path string to the directory where processed files will be output. |
| `landmarks_to_occlude` | `#!python list[list] or list[tuple]` | One or more lists of facial landmark paths. These paths can be manually created using `psyfaceutils.create_path()`, or you may use any of the library provided predefined landmark paths. |
| `occlusion_fill` | `#!python int` | An integer flag indicating the occlusion method to be used. One of `OCCLUSION_FILL_BLACK`, `OCCLUSION_FILL_MEAN` or `OCCLUSION_FILL_BAR`. |
| `with_sub_dirs`            | `#!python bool` | A boolean flag indicating if the input directory contains sub-directories. |
| `min_detection_confidence` | `#!python float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |
| `min_tracking_confidence`  | `#!python float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |
| `static_image_mode`        | `#!python bool` | A boolean flag indicating that the current filetype is static images. |

## Facial Colour Shifting

### face_color_shift( )

`face_color_shift` performs a weighted color shift on the color channel specified in `shift_color`, weighted by the outputs of the provided `timing_func`. Parameters `onset_t` and `offset_t` can be used to specify when the color shifting fades in and fades out (this only applies to video files). `face_color_shift` makes use of the CIELAB color space to perform color shifting, due to it being far more perceptually uniform than the standard RGB or BGR color spaces.

???+ note
    `face_color_shift` requires the outputs of the provided `timing_func` to be normalised; that is, in the range [0,1]. Predefined normalised functions such as `sigmoid`, `linear` and `gaussian` are available for use in `psyfaceutils`. Extra parameters for these functions can be passed to `face_color_shift` as keyword arguments. Users may also define their own timing functions, but it is up to the user to ensure their functions take at least one input float parameter, and that the return value is within the normal range.
    
```py
def face_color_shift(input_dir:str, output_dir:str, onset_t:float = 0.0, offset_t:float = 0.0, shift_magnitude: float = 8.0, timing_func:Callable[...,float] = sigmoid, shift_color:str|int = COLOR_RED, with_sub_dirs:bool = False, 
min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None: 
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir`                | `#!python str` | A path string to the directory containing files to process. |
| `output_dir`               | `#!python str` | A path string to the directory where processed files will be output. |
| `onset_t`                  | `#!python float` | The onset time when colour shifting will begin. |
| `offset_t`                 | `#!python float` | The offset time when colour shifting will begin to fade out. |
| `shift_magnitude`          | `#!python float` | The maximum units to shift the specified colour channel by, during peak onset. |
| `timing_func`              | `#!python Callable[..., float]` | Any function that takes at least one float, and returns a normalised float value. |
| `shift_colour`             | `#!python str, int` | Either a string literal (i.e. "red"), or a predefined integer constant; one of `COLOR_RED`, `COLOR_BLUE`, `COLOR_GREEN` or `COLOR_YELLOW`. |
| `with_sub_dirs`            | `#!python bool` | A boolean flag indicating if the input directory contains sub-directories. |
| `min_detection_confidence` | `#!python float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |
| `min_tracking_confidence`  | `#!python float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |

### face_saturation_shift( )

`face_saturation_shift` performs a weighted saturation shift in a near identical manner to `face_color_shift` above. This function makes use of the HSV color space to manipulate saturation. 

```py
def face_saturation_shift(input_dir:str, output_dir:str, onset_t:float = 0.0, offset_t:float = 0.0, shift_magnitude:float = -8.0, 
                          timing_func:Callable[..., float] = sigmoid, with_sub_dirs:bool = False, min_detection_confidence:float = 0.5,
                           min_tracking_confidence:float = 0.5) -> None:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir`                | `#!python str` | A path string to the directory containing files to process. |
| `output_dir`               | `#!python str` | A path string to the directory where processed files will be output. |
| `onset_t`                  | `#!python float` | The onset time when colour shifting will begin. |
| `offset_t`                 | `#!python float` | The offset time when colour shifting will begin to fade out. |
| `shift_magnitude`          | `#!python float` | The maximum units to shift the specified colour channel by, during peak onset. |
| `timing_func`              | `#!python Callable[..., float]` | Any function that takes at least one float, and returns a normalised float value. |
| `with_sub_dirs`            | `#!python bool` | A boolean flag indicating if the input directory contains sub-directories. |
| `min_detection_confidence` | `#!python float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |
| `min_tracking_confidence`  | `#!python float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |

### face_lightness_shift( )

`face_lightness_shift` performs a weighted lightness shift in a near identical manner to `face_color_shift` above. Lightness, represented by the L in the CIELAB color space, is taken as the cube root of the relative luminosity. Lightness ranges from 0-100, with black being defined as 0, and pure white being defined as 100.

!!! warning
    Lightness manipulation can cause unexpected behaviour when the initial lightness value is unknown. `face_lightness_shift` is currently experimental, so results may vary depending on the input files you are working with. Proceed with caution.

```py
def face_lightness_shift(input_dir:str, output_dir:str, onset_t:float = 0.0, offset_t:float = 0.0, shift_magnitude:float = 10.0, 
                        timing_func:Callable[..., float] = sigmoid, with_sub_dirs:bool = False, min_detection_confidence:float = 0.5,
                        min_tracking_confidence:float = 0.5) -> None:
```

| Parameter                  | Type           | Description                                               |
| :------------------------- | :------------- | :-------------------------------------------------------- |
| `input_dir`                | `#!python str` | A path string to the directory containing files to process. |
| `output_dir`               | `#!python str` | A path string to the directory where processed files will be output. |
| `onset_t`                  | `#!python float` | The onset time when colour shifting will begin. |
| `offset_t`                 | `#!python float` | The offset time when colour shifting will begin to fade out. |
| `shift_magnitude`          | `#!python float` | The maximum units to shift the specified colour channel by, during peak onset. |
| `timing_func`              | `#!python Callable[..., float]` | Any function that takes at least one float, and returns a normalised float value. |
| `with_sub_dirs`            | `#!python bool` | A boolean flag indicating if the input directory contains sub-directories. |
| `min_detection_confidence` | `#!python float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |
| `min_tracking_confidence`  | `#!python float` | A confidence measure in the range [0,1], passed on to the MediaPipe FaceMesh model. |