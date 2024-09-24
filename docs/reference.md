# API Reference

## Facial Masking


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


??? info
    Parameters `min_detection_confidence`, `min_tracking_confidence`, and `static_image_mode` are passed directly to the declaration of the MediaPipe FaceMesh model. 

    ``` py
    import mediapipe as mp

    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, min_detection_confidence = min_detection_confidence,
                                        min_tracking_confidence = min_tracking_confidence, static_image_mode = static_image_mode)
    ```

    For more information on MediaPipes FaceMesh solution, [see here.](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)