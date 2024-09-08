# Code Reference

## Dynamic Facial Masking
``` py
def mask_face_region(input_dir:str, output_dir:str, mask_type:int = FACE_SKIN_ISOLATION, with_sub_dirs:bool = False,
                     min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5, static_image_mode:bool = False) -> None:
    """Applies specified mask type to video or image files located in input_dir, then outputs masked files to output_dir.

    Parameters
    ----------

    input_dir: str
        A path string of the directory containing files to process.

    output_dir: str
        A path string of the directory where processed videos and images will be written to.

    mask_type: int
        An integer indicating the type of mask to apply to the input videos. This can be one of three mask configuration options: FACE_OVAL, FACE_OVAL_TIGHT, FACE_SKIN_ISOLATION

    with_sub_dirs: bool
        Indicates if the input directory contains nested subdirectories.

    min_detection_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
    
    min_tracking_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
    
    static_image_mode: bool
        A boolean flag indicating to the mediapipe FaceMesh that it is working with static images rather than
        video frames.
    
    Raises
    ------

    ValueError 
        Given an unknown mask type.
    TypeError 
        Given invalid parameter types.
    OSError: 
        Given invalid path strings for in/output directories
    """
```