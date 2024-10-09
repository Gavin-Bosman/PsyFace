import cv2 as cv
import cv2.typing
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import sys
from typing import Callable
from .psyfaceutils import *
from operator import itemgetter

def mask_face_region(input_dir:str, output_dir:str, mask_type:int = FACE_OVAL_MASK, with_sub_dirs:bool = False,
                     min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5, static_image_mode:bool = False) -> None:
    """Applies specified mask type to video files located in input_dir, then outputs masked videos to output_dir.

    Parameters
    ----------

    input_dir: str
        A path string of the directory containing videos to process.

    output_dir: str
        A path string of the directory where processed videos will be written to.

    mask_type: int
        An integer indicating the type of mask to apply to the input videos. This can be one of four options:
        FACE_OVAL_MASK, FACE_OVAL_TIGHT_MASK, FACE_SKIN_MASK or EYES_NOSE_MOUTH_MASK.

    with_sub_dirs: bool
        Indicates if the input directory contains subfolders.

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

    global MASK_OPTIONS
    global FACE_OVAL_MASK
    global FACE_OVAL_TIGHT_MASK
    global FACE_SKIN_MASK
    global EYES_NOSE_MOUTH_MASK

    global LEFT_EYE_PATH
    global RIGHT_EYE_PATH
    global LIPS_PATH
    global NOSE_PATH
    global FACE_OVAL_PATH
    global FACE_OVAL_TIGHT_PATH
    singleFile = False

    def process_frame(frame: cv.typing.MatLike, mask_type: int) -> cv.typing.MatLike:
        match mask_type:
            case 14: # Eyes nose mouth mask
                face_mesh_results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                landmark_screen_coords = []

                if face_mesh_results.multi_face_landmarks:
                    for face_landmarks in face_mesh_results.multi_face_landmarks:

                        # Convert normalised landmark coordinates to x-y pixel coordinates
                        for id,lm in enumerate(face_landmarks.landmark):
                            ih, iw, ic = frame.shape
                            x,y = int(lm.x * iw), int(lm.y * ih)
                            landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
                else:
                    print("Mask_face_region: Face mesh detection error.")
                    sys.exit(1)

                le_screen_coords = []
                re_screen_coords = []
                nose_screen_coords = []
                lips_screen_coords = []

                 # left eye screen coordinates
                for cur_source, cur_target in LEFT_EYE_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    le_screen_coords.append((source.get('x'),source.get('y')))
                    le_screen_coords.append((target.get('x'),target.get('y')))
                
                # right eye screen coordinates
                for cur_source, cur_target in RIGHT_EYE_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    re_screen_coords.append((source.get('x'),source.get('y')))
                    re_screen_coords.append((target.get('x'),target.get('y')))
                
                # nose screen coordinates
                for cur_source, cur_target in NOSE_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    nose_screen_coords.append((source.get('x'),source.get('y')))
                    nose_screen_coords.append((target.get('x'),target.get('y')))

                # lips screen coordinates
                for cur_source, cur_target in LIPS_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    lips_screen_coords.append((source.get('x'),source.get('y')))
                    lips_screen_coords.append((target.get('x'),target.get('y')))

                # Creating boolean masks for the facial regions
                le_mask = np.zeros((frame.shape[0],frame.shape[1]))
                le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
                le_mask = le_mask.astype(bool)

                re_mask = np.zeros((frame.shape[0],frame.shape[1]))
                re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
                re_mask = re_mask.astype(bool)

                nose_mask = np.zeros((frame.shape[0], frame.shape[1]))
                nose_mask = cv.fillConvexPoly(nose_mask, np.array(nose_screen_coords), 1)
                nose_mask = nose_mask.astype(bool)

                lip_mask = np.zeros((frame.shape[0],frame.shape[1]))
                lip_mask = cv.fillConvexPoly(lip_mask, np.array(lips_screen_coords), 1)
                lip_mask = lip_mask.astype(bool)

                masked_frame = np.zeros((frame.shape[0],frame.shape[1],1), dtype=np.uint8)
                masked_frame[le_mask] = 255
                masked_frame[re_mask] = 255
                masked_frame[nose_mask] = 255
                masked_frame[lip_mask] = 255

                masked_frame = np.where(masked_frame == 255, frame, masked_frame)
                return masked_frame

            case 3: # Face skin isolation
                face_mesh_results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                landmark_screen_coords = []

                if face_mesh_results.multi_face_landmarks:
                    for face_landmarks in face_mesh_results.multi_face_landmarks:

                        # Convert normalised landmark coordinates to x-y pixel coordinates
                        for id,lm in enumerate(face_landmarks.landmark):
                            ih, iw, ic = frame.shape
                            x,y = int(lm.x * iw), int(lm.y * ih)
                            landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
                else:
                    print("Mask_face_region: Face mesh detection error.")
                    sys.exit(1)

                le_screen_coords = []
                re_screen_coords = []
                lips_screen_coords = []
                face_outline_coords = []

                # left eye screen coordinates
                for cur_source, cur_target in LEFT_EYE_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    le_screen_coords.append((source.get('x'),source.get('y')))
                    le_screen_coords.append((target.get('x'),target.get('y')))
                
                # right eye screen coordinates
                for cur_source, cur_target in RIGHT_EYE_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    re_screen_coords.append((source.get('x'),source.get('y')))
                    re_screen_coords.append((target.get('x'),target.get('y')))

                # lips screen coordinates
                for cur_source, cur_target in LIPS_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    lips_screen_coords.append((source.get('x'),source.get('y')))
                    lips_screen_coords.append((target.get('x'),target.get('y')))
                
                # face oval screen coordinates
                for cur_source, cur_target in FACE_OVAL_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    face_outline_coords.append((source.get('x'),source.get('y')))
                    face_outline_coords.append((target.get('x'),target.get('y')))

                # Creating boolean masks for the facial regions
                le_mask = np.zeros((frame.shape[0],frame.shape[1]))
                le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
                le_mask = le_mask.astype(bool)

                re_mask = np.zeros((frame.shape[0],frame.shape[1]))
                re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
                re_mask = re_mask.astype(bool)

                lip_mask = np.zeros((frame.shape[0],frame.shape[1]))
                lip_mask = cv.fillConvexPoly(lip_mask, np.array(lips_screen_coords), 1)
                lip_mask = lip_mask.astype(bool)

                oval_mask = np.zeros((frame.shape[0],frame.shape[1]))
                oval_mask = cv.fillConvexPoly(oval_mask, np.array(face_outline_coords), 1)
                oval_mask = oval_mask.astype(bool)

                # Otsu thresholding to seperate foreground and background
                grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                grey_blurred = cv.GaussianBlur(grey_frame, (7,7), 0)
                thresh_val, thresholded = cv.threshold(grey_blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

                # Adding a temporary image border to allow for correct floodfill behaviour
                bordered_thresholded = cv.copyMakeBorder(thresholded, 10, 10, 10, 10, cv.BORDER_CONSTANT)
                floodfilled = bordered_thresholded.copy()
                cv.floodFill(floodfilled, None, (0,0), 255)

                # Removing temporary border and creating foreground mask
                floodfilled = floodfilled[10:-10, 10:-10]
                floodfilled = cv.bitwise_not(floodfilled)
                foreground = cv.bitwise_or(thresholded, floodfilled)

                # Masking the face oval
                masked_frame = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                masked_frame[oval_mask] = 255

                # Masking out eyes and lips
                masked_frame[le_mask] = 0
                masked_frame[re_mask] = 0
                masked_frame[lip_mask] = 0

                # Last step, masking out the bounding face shape
                masked_frame = cv.bitwise_and(masked_frame, foreground)
                masked_frame = np.reshape(masked_frame, (masked_frame.shape[0], masked_frame.shape[1], 1))
                frame = cv.bitwise_and(frame, frame, mask = foreground)
                masked_frame = np.where(masked_frame == 255, frame, 255)
                # masked_frame = np.where(masked_frame == 255, frame, masked_frame) for black background in output
                return masked_frame

            case 2: # Face oval tight
                face_mesh_results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                landmark_screen_coords = []

                if face_mesh_results.multi_face_landmarks:
                    for face_landmarks in face_mesh_results.multi_face_landmarks:

                        # Convert normalised landmark coordinates to x-y pixel coordinates
                        for id,lm in enumerate(face_landmarks.landmark):
                            ih, iw, ic = frame.shape
                            x,y = int(lm.x * iw), int(lm.y * ih)
                            landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
                else:
                    print("Mask_face_region: Face mesh detection error.")
                    sys.exit(1)

                face_outline_coords = []
                
                # face oval screen coordinates
                for cur_source, cur_target in FACE_OVAL_TIGHT_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    face_outline_coords.append((source.get('x'),source.get('y')))
                    face_outline_coords.append((target.get('x'),target.get('y')))

                oval_mask = np.zeros((frame.shape[0],frame.shape[1]))
                oval_mask = cv.fillConvexPoly(oval_mask, np.array(face_outline_coords), 1)
                oval_mask = oval_mask.astype(bool)

                # Otsu thresholding to seperate foreground and background
                grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                grey_blurred = cv.GaussianBlur(grey_frame, (7,7), 0)
                thresh_val, thresholded = cv.threshold(grey_blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

                # Adding a temporary image border to allow for correct floodfill behaviour
                bordered_thresholded = cv.copyMakeBorder(thresholded, 10, 10, 10, 10, cv.BORDER_CONSTANT)
                floodfilled = bordered_thresholded.copy()
                cv.floodFill(floodfilled, None, (0,0), 255)

                # Removing temporary border and creating foreground mask
                floodfilled = floodfilled[10:-10, 10:-10]
                floodfilled = cv.bitwise_not(floodfilled)
                foreground = cv.bitwise_or(thresholded, floodfilled)

                # Masking the face oval
                masked_frame = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                masked_frame[oval_mask] = 255

                # Last step, masking out the bounding face shape
                masked_frame = np.reshape(masked_frame, (masked_frame.shape[0], masked_frame.shape[1], 1))
                frame = cv.bitwise_and(frame, frame, foreground)
                masked_frame = np.where(masked_frame == 255, frame, 255)

                return masked_frame
            
            case 1: # Face oval
                face_mesh_results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                landmark_screen_coords = []

                if face_mesh_results.multi_face_landmarks:
                    for face_landmarks in face_mesh_results.multi_face_landmarks:

                        # Convert normalised landmark coordinates to x-y pixel coordinates
                        for id,lm in enumerate(face_landmarks.landmark):
                            ih, iw, ic = frame.shape
                            x,y = int(lm.x * iw), int(lm.y * ih)
                            landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
                else:
                    print("Mask_face_region: Face mesh detection error.")
                    sys.exit(1)

                face_outline_coords = []
                
                # face oval screen coordinates
                for cur_source, cur_target in FACE_OVAL_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    face_outline_coords.append((source.get('x'),source.get('y')))
                    face_outline_coords.append((target.get('x'),target.get('y')))

                oval_mask = np.zeros((frame.shape[0],frame.shape[1]))
                oval_mask = cv.fillConvexPoly(oval_mask, np.array(face_outline_coords), 1)
                oval_mask = oval_mask.astype(bool)

                # Otsu thresholding to seperate foreground and background
                grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                grey_blurred = cv.GaussianBlur(grey_frame, (7,7), 0)
                thresh_val, thresholded = cv.threshold(grey_blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

                # Adding a temporary image border to allow for correct floodfill behaviour
                bordered_thresholded = cv.copyMakeBorder(thresholded, 10, 10, 10, 10, cv.BORDER_CONSTANT)
                floodfilled = bordered_thresholded.copy()
                cv.floodFill(floodfilled, None, (0,0), 255)

                # Removing temporary border and creating foreground mask
                floodfilled = floodfilled[10:-10, 10:-10]
                floodfilled = cv.bitwise_not(floodfilled)
                foreground = cv.bitwise_or(thresholded, floodfilled)

                # Masking the face oval
                masked_frame = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                masked_frame[oval_mask] = 255

                # Last step, masking out the bounding face shape
                masked_frame = np.reshape(masked_frame, (masked_frame.shape[0], masked_frame.shape[1], 1))
                frame = cv.bitwise_and(frame, frame, foreground)
                masked_frame = np.where(masked_frame == 255, frame, 255)

                return masked_frame
            
            case _:
                print("Mask_face_region: Undefined facial mask, please specify one of FACE_SKIN_ISOLATION, FACE_OVAL, FACE_OVAL_TIGHT.")
                sys.exit(1)
            
    # Type and value checks for function parameters
    if not isinstance(input_dir, str):
        raise TypeError("Mask_face_region: parameter input_dir must be of type str.")
    elif not os.path.exists(input_dir):
        raise OSError("Mask_face_region: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(input_dir):
        singleFile = True
    
    if not isinstance(output_dir, str):
        raise TypeError("Mask_face_region: parameter output_dir must be of type str.")
    elif not os.path.exists(output_dir):
        raise ValueError("Mask_face_region: output directory path is not a valid path, or the directory does not exist.")
    
    if mask_type not in MASK_OPTIONS:
        raise ValueError("Mask_face_region: mask_type must be either 1: indicating FACE_OVAL, 2: indicating FACE_OVAL_TIGHT, or 3: indicating FACE_SKIN_ISOLATION.")
    
    if not isinstance(with_sub_dirs, bool):
        raise TypeError("Mask_face_region: parameter with_sub_dirs must be of type bool.")
    
    if not isinstance(min_detection_confidence, float):
        raise TypeError("Mask_face_region: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        raise ValueError("Mask_face_region: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        raise TypeError("Mask_face_region: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        raise ValueError("Mask_face_region: parameter min_tracking_confidence must be in the range [0,1].")
    
    if not isinstance(static_image_mode, bool):
        raise TypeError("Mask_face_region: parameter static_image_mode must be of type bool.")
    
    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = static_image_mode, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)

    # Creating a list of file names to iterate through when processing
    files_to_process = []
    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    # Creating named output directories for video output
    if not os.path.isdir(output_dir + "\\Masked_Video_Output"):
        os.mkdir(output_dir + "\\Masked_Video_Output")
    output_dir = output_dir + "\\Masked_Video_Output"

    for file in files_to_process:

        # Sniffing input filetype to determine running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".jpg" | ".jpeg" | ".png" | ".bmp":
                static_image_mode = True
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = True, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case _:
                print("Mask_face_region: Incompatible video or image file type. Please see utils.transcode_video_to_mp4().")
                sys.exit(1)

        capture = None
        result = None
        
        if not static_image_mode:

            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                print("Mask_face_region: Error opening VideoCapture object.")
                sys.exit(1)
            
            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(output_dir + "\\" + filename + "_masked" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                print("Mask_face_region: Error opening VideoWriter object.")
                sys.exit(1)
            
            while True:
                success, frame = capture.read()
                if not success:
                    break
                
                masked_frame = process_frame(frame, mask_type)
                result.write(masked_frame)
        
            capture.release()
            result.release()
        
        else:
            img = cv.imread(file)
            masked_img = process_frame(img, mask_type)
            success = cv.imwrite(output_dir + "\\" + filename + "_masked" + extension, masked_img)
            if not success:
                print("Mask_face_region: cv2.imwrite error.")
                sys.exit(1)

def occlude_face_region(input_dir:str, output_dir:str, landmarks_to_occlude:list[list[tuple]] | list[tuple], occlusion_fill:int = OCCLUSION_FILL_BLACK,
                        with_sub_dirs:bool =  False, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    ''' For each video or image contained within the input directory, the landmark regions contained within landmarks_to_occlude 
    will be occluded with either black or the facial mean pixel value. Processed files are then output to Occluded_Video_Output 
    within the specified output directory.

    Parameters
    ----------

    input_dir: str
        A path string to the directory containing files to process. 

    output_dir: str
        A path string to the directory where processed videos will be written.

    landmarks_to_occlude: list of list, list of tuple
        A list of facial landmark paths, either created by the user using utils.create_path(), or selected from the 
        predefined set of facial landmark paths.
    
    occlusion_fill: int
        An integer flag indicating the fill method of the occluded landmark regions. One of OCCLUSION_FILL_BLACK or 
        OCCLUSION_FILL_MEAN.
    
    with_sub_dirs: bool
        A boolean flag indicating if the input directory contains subfolders.
    
    min_detection_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.

    min_tracking_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
        
    Raises
    ------

    TypeError 
        Given invalid parameter types.
    ValueError 
        Given invalid landmark sets or unrecognized fill options.
    OSError 
        Given invalid path strings to either input_dir or output_dir.
    '''
    
    global OCCLUSION_FILL_BLACK
    global OCCLUSION_FILL_MEAN
    global OCCLUSION_FILL_BAR
    global FACE_OVAL_TIGHT_PATH
    global FACE_OVAL_PATH
    singleFile = False
    static_image_mode = False

    def occlude_frame(frame:cv.typing.MatLike, mask:cv.typing.MatLike, occlusion_fill:int) -> cv.typing.MatLike:
        match occlusion_fill:
            case 8:
                masked_frame = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
                frame = np.where(masked_frame == 255, 0, frame)
                return frame

            case 9:
                cur_landmark_coords = []
                for cur_source, cur_target in FACE_OVAL_TIGHT_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    cur_landmark_coords.append((source.get('x'),source.get('y')))
                    cur_landmark_coords.append((target.get('x'),target.get('y')))

                # Creating boolean masks for the facial landmarks 
                bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                bool_mask = cv.fillConvexPoly(bool_mask, np.array(cur_landmark_coords), 1)
                bool_mask = bool_mask.astype(bool)

                # Extracting the mean pixel value of the face
                bin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                bin_mask[bool_mask] = 255
                mean = cv.mean(frame, bin_mask)
                
                # Fill occlusion regions with facial mean
                masked_frame = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
                mean_img = np.zeros_like(frame, dtype=np.uint8)
                mean_img[:] = mean[:3]
                frame = np.where(masked_frame == 255, mean_img, frame)
                return frame
            
            case 10:
                if LEFT_EYE_PATH in landmarks_to_occlude or RIGHT_EYE_PATH in landmarks_to_occlude:
                    start = landmark_screen_coords[21]
                    start = (start.get('x'), start.get('y'))
                    end = landmark_screen_coords[447]
                    end = (end.get('x'), end.get('y'))
                   
                    frame = cv2.rectangle(frame, start, end, (0,0,0), -1)
                
                elif LIPS_PATH in landmarks_to_occlude:
                    start = landmark_screen_coords[213]
                    start = (start.get('x') -10, start.get('y'))
                    end = landmark_screen_coords[365]
                    end = (end.get('x') + 20, end.get('y') + 20)

                    frame = cv2.rectangle(frame, start, end, (0,0,0), -1)

                elif NOSE_PATH in landmarks_to_occlude:
                    start = landmark_screen_coords[227]
                    start = (start.get('x'), start.get('y'))
                    end = landmark_screen_coords[433]
                    end = (end.get('x') + 20, end.get('y'))

                    frame = cv2.rectangle(frame, start, end, (0,0,0), -1)
                
                else:
                    print("""Occlude_face_region: OCCLUSION_FILL_BAR is only compatible with EYE, LIPS and NOSE paths.
                          Please check your provided landmark paths and try again.""")
                    sys.exit(1)
                
                return frame

    # Performing checks on function parameters
    if not isinstance(input_dir, str):
        raise TypeError("Occlude_face_region: invalid type for parameter input_dir.")
    elif not os.path.exists(input_dir):
        raise OSError("Occlude_face_region: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(input_dir):
        singleFile = True
    
    if not isinstance(output_dir, str):
        raise TypeError("Occlude_face_region: parameter output_dir must be a str.")
    elif not os.path.exists(output_dir):
        raise OSError("Occlude_face_region: output directory path is not a valid path, or the directory does not exist.")
    elif not os.path.isdir(output_dir):
        raise ValueError("Occlude_face_region: output_dir must be a valid path to a directory.")
    
    if not isinstance(landmarks_to_occlude, list):
        raise TypeError("Occlude_face_region: parameter landmarks_to_occlude expects a list.")
    if not isinstance(landmarks_to_occlude[0], list) or isinstance(landmarks_to_occlude[0], tuple):
        raise ValueError("Occlude_face_region: landmarks_to_occlude may either be a list of lists, or a singular list of tuples.")
    
    if not isinstance(occlusion_fill, int):
        raise TypeError("Occlude_face_region: parameter occlusion_fill must be of type int.")
    elif occlusion_fill not in [OCCLUSION_FILL_BLACK, OCCLUSION_FILL_MEAN, OCCLUSION_FILL_BAR]:
        raise ValueError("Occlude_face_region: parameter occlusion_fill must be one of OCCLUSION_FILL_BLACK, OCCLUSION_FILL_MEAN or OCCLUSION_FILL_BAR.")
    
    if not isinstance(with_sub_dirs, bool):
        raise TypeError("Occlude_face_region: parameter with_sub_dirs must be of type bool.")
    
    if not isinstance(min_detection_confidence, float):
        raise TypeError("Occlude_face_region: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        raise ValueError("Occlude_face_region: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        raise TypeError("Occlude_face_region: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        raise ValueError("Occlude_face_region: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = static_image_mode, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)

    # Creating a list of file names to iterate through when processing
    files_to_process = []
    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    if not os.path.exists(output_dir + "\\Occluded_Video_Output"):
        os.mkdir(output_dir + "\\Occluded_Video_Output")
    output_dir = output_dir + "\\Occluded_Video_Output"

    for file in files_to_process:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".jpg" | ".jpeg" | ".png" | ".bmp":
                static_image_mode = True
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = True, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case _:
                print("Occlude_face_region: Incompatible video or image file type. Please see utils.transcode_video_to_mp4().")
                sys.exit(1)

        capture = None
        result = None

        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                print("Occlude_face_region: Error opening video file.")
                sys.exit(1)

            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(output_dir + "\\" + filename + "_occluded" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                print("Occlude_face_region: Error opening VideoWriter object.")
                sys.exit(1)

        while True:

            if static_image_mode:
                frame = cv.imread(file)
            else:
                success, frame = capture.read()
                if not success:
                    break    

            face_mesh_results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            landmark_screen_coords = []

            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:

                    # Convert normalised landmark coordinates to x-y pixel coordinates
                    for id,lm in enumerate(face_landmarks.landmark):
                        ih, iw, ic = frame.shape
                        x,y = int(lm.x * iw), int(lm.y * ih)
                        landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
            else:
                if static_image_mode:
                    print("Occlude_face_region: face mesh detection error.")
                    sys.exit(1)
                else: 
                    continue

            masked_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            # Creating masks for occluded regions
            if isinstance(landmarks_to_occlude[0], list):
                for landmark_set in landmarks_to_occlude:

                    face_oval_coords = []
                    # Converting landmark coords to screen coords
                    for cur_source, cur_target in FACE_OVAL_PATH:
                        source = landmark_screen_coords[cur_source]
                        target = landmark_screen_coords[cur_target]
                        face_oval_coords.append((source.get('x'),source.get('y')))
                        face_oval_coords.append((target.get('x'),target.get('y')))
                    
                    # Creating boolean masks for the face oval
                    face_oval_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    face_oval_mask = cv.fillConvexPoly(face_oval_mask, np.array(face_oval_coords), 1)
                    face_oval_mask = face_oval_mask.astype(bool)

                    max_x = max(face_oval_coords, key=itemgetter(0))[0]
                    min_x = min(face_oval_coords, key=itemgetter(0))[0]

                    max_y = max(face_oval_coords, key=itemgetter(1))[1]
                    min_y = min(face_oval_coords, key=itemgetter(1))[1]

                    # Compute the center bisecting lines of the face oval
                    cx = round((max_y + min_y)/2)           
                    cy = round((max_x + min_x)/2)

                    match landmark_set:
                        # Horizontal hemi face top:
                        case [(0,)]:

                            oval_masked_img = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                            oval_masked_img[face_oval_mask] = 255

                            split_line = cx
                            poly = []

                            for i in range(len(face_oval_coords)):
                                p1 = face_oval_coords[i]
                                p2 = face_oval_coords[(i + 1) % len(face_oval_coords)]

                                if (p1[1] < split_line):
                                    poly.append(p1)
                                
                                intersection = compute_line_intersection(p1, p2, split_line, False)

                                if intersection != None:
                                    poly.append(intersection)
                            
                            hemi_face_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                            hemi_face_mask = cv.fillConvexPoly(hemi_face_mask, np.array(poly), 1)
                            hemi_face_mask = hemi_face_mask.astype(bool)

                            masked_frame[hemi_face_mask] = 255
                            break

                        # Horizontal hemi face bottom:
                        case [(1,)]:
                            oval_masked_img = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                            oval_masked_img[face_oval_mask] = 255

                            split_line = cx
                            poly = []

                            for i in range(len(face_oval_coords)):
                                p1 = face_oval_coords[i]
                                p2 = face_oval_coords[(i + 1) % len(face_oval_coords)]

                                if (p1[1] > split_line):
                                    poly.append(p1)
                                
                                intersection = compute_line_intersection(p1, p2, split_line, False)

                                if intersection != None:
                                    poly.append(intersection)
                            
                            hemi_face_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                            hemi_face_mask = cv.fillConvexPoly(hemi_face_mask, np.array(poly), 1)
                            hemi_face_mask = hemi_face_mask.astype(bool)

                            masked_frame[hemi_face_mask] = 255
                            break

                        # Vertical hemi face left:
                        case [(2,)]:
                            oval_masked_img = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                            oval_masked_img[face_oval_mask] = 255

                            split_line = cy
                            poly = []

                            for i in range(len(face_oval_coords)):
                                p1 = face_oval_coords[i]
                                p2 = face_oval_coords[(i + 1) % len(face_oval_coords)]

                                if (p1[0] < split_line):
                                    poly.append(p1)
                                
                                intersection = compute_line_intersection(p1, p2, split_line, True)

                                if intersection != None:
                                    poly.append(intersection)
                            
                            hemi_face_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                            hemi_face_mask = cv.fillConvexPoly(hemi_face_mask, np.array(poly), 1)
                            hemi_face_mask = hemi_face_mask.astype(bool)

                            masked_frame[hemi_face_mask] = 255
                            break

                        # Vertical hemi face right
                        case [(3,)]:
                            oval_masked_img = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                            oval_masked_img[face_oval_mask] = 255

                            split_line = cy
                            poly = []

                            for i in range(len(face_oval_coords)):
                                p1 = face_oval_coords[i]
                                p2 = face_oval_coords[(i + 1) % len(face_oval_coords)]

                                if (p1[0] > split_line):
                                    poly.append(p1)
                                
                                intersection = compute_line_intersection(p1, p2, split_line, True)

                                if intersection != None:
                                    poly.append(intersection)
                            
                            hemi_face_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                            hemi_face_mask = cv.fillConvexPoly(hemi_face_mask, np.array(poly), 1)
                            hemi_face_mask = hemi_face_mask.astype(bool)

                            masked_frame[hemi_face_mask] = 255
                            break
                        case _:
                            continue

                    cur_landmark_coords = []
                    # Converting landmark coords to screen coords
                    for cur_source, cur_target in landmark_set:
                        source = landmark_screen_coords[cur_source]
                        target = landmark_screen_coords[cur_target]
                        cur_landmark_coords.append((source.get('x'),source.get('y')))
                        cur_landmark_coords.append((target.get('x'),target.get('y')))
                    
                    # Creating boolean masks for the facial landmarks 
                    bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                    bool_mask = cv.fillConvexPoly(bool_mask, np.array(cur_landmark_coords), 1)
                    bool_mask = bool_mask.astype(bool)

                    masked_frame[bool_mask] = 255
            else:
                cur_landmark_coords = []
                # Converting landmark coords to screen coords
                for cur_source, cur_target in landmarks_to_occlude:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    cur_landmark_coords.append((source.get('x'),source.get('y')))
                    cur_landmark_coords.append((target.get('x'),target.get('y')))
                
                # Creating boolean masks for the facial landmarks 
                bool_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
                bool_mask = cv.fillConvexPoly(bool_mask, np.array(cur_landmark_coords), 1)
                bool_mask = bool_mask.astype(bool)

                masked_frame[bool_mask] = 255
            
            frame = occlude_frame(frame, masked_frame, occlusion_fill)

            if static_image_mode:
                cv.imwrite(output_dir + "\\" + filename + "_occluded" + extension, frame)
                break
            else:
                result.write(frame)

        if not static_image_mode:
            capture.release()
            result.release()

def blur_face_region(input_dir:str, output_dir:str, blur_method:str | int = "gaussian", k_size:int = 15, with_sub_dirs:bool = False, 
                     min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    """ For each video or image file within `input_dir`, the specified `blur_method` will be applied. Blurred images and video files
    are then written out to `output_dir`.

    Parameters
    ----------

    input_dir: str
        A path string to a directory containing the image or video files to be processed.

    output_dir: str
        A path string to a directory where processed files will be written.

    blur_method: str, int
        Either a string literal ("average", "gaussian", "median"), or a predefined integer constant 
        (BLUR_METHOD_AVERAGE, BLUR_METHOD_GAUSSIAN, BLUR_METHOD_MEDIAN) specifying the type of blurring operation to be performed.
    
    k_size: int
        Specifies the size of the square kernel used in blurring operations. 
    
    with_sub_dirs: bool
        A boolean flag indicating if the input directory contains nested sub-directories.
    
    min_detection_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.

    min_tracking_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
    
    Raises
    ------

    TypeError:
        Given invalid or incompatible input parameter types.
    ValueError:
        Given an unrecognized value.
    OSError:
        Given invalid path strings to input or output directory.

    """

    global BLUR_METHOD_AVERAGE
    global BLUR_METHOD_GAUSSIAN
    global BLUR_METHOD_MEDIAN
    global FACE_OVAL_PATH
    singlefile = False
    static_image_mode = False

    # Performing checks on function parameters
    if not isinstance(input_dir, str):
        raise TypeError("Blur_face_region: invalid type for parameter input_dir.")
    elif not os.path.exists(input_dir):
        raise OSError("Blur_face_region: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(input_dir):
        singleFile = True
    
    if not isinstance(output_dir, str):
        raise TypeError("Blur_face_region: parameter output_dir must be a str.")
    elif not os.path.exists(output_dir):
        raise OSError("Blur_face_region: output directory path is not a valid path, or the directory does not exist.")
    elif not os.path.isdir(output_dir):
        raise ValueError("Blur_face_region: output_dir must be a valid path to a directory.")
    
    if isinstance(blur_method, str):
        if blur_method not in ["average", "Average", "gaussian", "Gaussian", "median", "Median"]:
            raise ValueError("Blur_face_region: Unrecognised value for parameter blur_method.")
    elif isinstance(blur_method, int):
        if blur_method not in [BLUR_METHOD_AVERAGE, BLUR_METHOD_GAUSSIAN, BLUR_METHOD_MEDIAN]:
            raise ValueError("Blur_face_region: Unrecognised value for parameter blur_method.")
    else:
        raise TypeError("Blur_face_region: Incompatable type for parameter blur_method.")
    
    if not isinstance(k_size, int):
        raise TypeError("Blur_face_region: parameter k_size must be of type int.")
    
    if not isinstance(with_sub_dirs, bool):
        raise TypeError("Blur_face_region: parameter with_sub_dirs must be of type bool.")
    
    if not isinstance(min_detection_confidence, float):
        raise TypeError("Blur_face_region: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        raise ValueError("Blur_face_region: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        raise TypeError("Blur_face_region: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        raise ValueError("Blur_face_region: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Defining the mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = static_image_mode, min_detection_confidence = min_detection_confidence,
                                                min_tracking_confidence = min_tracking_confidence)

    # Creating a list of file names to iterate through when processing
    files_to_process = []
    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    if not os.path.exists(output_dir + "\\Blurred"):
        os.mkdir(output_dir + "\\Blurred")
    output_dir = output_dir + "\\Blurred"

    for file in files_to_process:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".jpg" | ".jpeg" | ".png" | ".bmp":
                static_image_mode = True
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = True, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case _:
                print("Blur_face_region: Incompatible video or image file type. Please see utils.transcode_video_to_mp4().")
                sys.exit(1)

        capture = None
        result = None

        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                print("Blur_face_region: Error opening video file.")
                sys.exit(1)

            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(output_dir + "\\" + filename + "_blurred" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                print("Blur_face_region: Error opening VideoWriter object.")
                sys.exit(1)

        while True:

            if static_image_mode:
                frame = cv.imread(file)
            else:
                success, frame = capture.read()
                if not success:
                    break    

            face_mesh_results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            landmark_screen_coords = []

            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:

                    # Convert normalised landmark coordinates to x-y pixel coordinates
                    for id,lm in enumerate(face_landmarks.landmark):
                        ih, iw, ic = frame.shape
                        x,y = int(lm.x * iw), int(lm.y * ih)
                        landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
            else:
                if static_image_mode:
                    print("Blur_face_region: face mesh detection error.")
                    sys.exit(1)
                else: 
                    continue

            face_outline_coords = []
            # face oval screen coordinates
            for cur_source, cur_target in FACE_OVAL_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                face_outline_coords.append((source.get('x'),source.get('y')))
                face_outline_coords.append((target.get('x'),target.get('y')))
            
            oval_mask = np.zeros((frame.shape[0],frame.shape[1]))
            oval_mask = cv.fillConvexPoly(oval_mask, np.array(face_outline_coords), 1)
            oval_mask = oval_mask.astype(bool)

            masked_frame = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
            masked_frame[oval_mask] = 255

            frame_blurred = None

            match blur_method:
                case "average" | "Average":
                    frame_blurred = cv.blur(frame, (k_size, k_size))
                    frame = np.where(masked_frame == 255, frame_blurred, frame)
                case 11:
                    frame_blurred = cv.blur(frame, (k_size, k_size))
                    frame = np.where(masked_frame == 255, frame_blurred, frame)
                
                case "gaussian" | "Gaussian":
                    frame_blurred = cv.GaussianBlur(frame, (k_size, k_size), 0)
                    frame = np.where(masked_frame == 255, frame_blurred, frame)
                case 12:
                    frame_blurred = cv.GaussianBlur(frame, (k_size, k_size), 0)
                    frame = np.where(masked_frame == 255, frame_blurred, frame)
                
                case "median" | "Median":
                    frame_blurred = cv.medianBlur(frame, k_size)
                    frame = np.where(masked_frame == 255, frame_blurred, frame)
                case 13:
                    frame_blurred = cv.medianBlur(frame, k_size)
                    frame = np.where(masked_frame == 255, frame_blurred, frame)
                
                case _:
                    print("Blur_face_region: Unrecognised value for parameter blur_method.")
                    sys.exit(1)
                    

            if static_image_mode:
                cv.imwrite(output_dir + "\\" + filename + "_blurred" + extension, frame)
                break
            else:
                result.write(frame)

        if not static_image_mode:
            capture.release()
            result.release()

# Expand color extraction by allowing custom masks to be passed, would allow dynamic colour extraction from specified regions

def extract_color_channel_means(input_dir:str, output_dir:str, color_space: int|str = COLOR_SPACE_RGB, with_sub_dirs:bool = False, 
                                mask_face:bool = True, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None:
    """Extracts and outputs mean values of each color channel from the specified color space. Creates a new directory 
    'CSV_Output', where a csv file will be written for each input video file provided.

    Parameters
    ----------

    input_dir: str
        A path string to a directory containing the video files to be processed.

    output_dir: str
        A path string to a directory where outputted csv files will be written to.
    
    color_space: int, str
        A specifier for which color space to operate in.
    
    with_sub_dirs: bool
        Indicates whether the input directory contains subfolders.
    
    mask_face: bool
        Indicates whether to mask the face region prior to extracting color means.
    
    min_detection_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.

    min_tracking_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
        
    Raises
    ------

    TypeError
        Given invalid parameter types.
    ValueError 
        Given an unrecognized color space.
    OSError 
        If input or output directories are invalid paths.
    """
    
    # Global declarations and init
    global COLOR_SPACE_RGB
    global COLOR_SPACE_HSV
    global COLOR_SPACE_GRAYSCALE
    global FACE_OVAL_TIGHT_PATH
    singleFile = False

    # Type and value checking input parameters
    if not isinstance(input_dir, str):
        raise TypeError("Extract_color_channel_means: input_dir must be a path string.")
    elif not os.path.exists(input_dir):
        raise OSError("Extract_color_channel_means: input_dir is not a valid path.")
    elif os.path.isfile(input_dir):
        singleFile = True
    
    if not isinstance(output_dir, str):
        raise TypeError("Extract_color_channel_means: output_dir must be a path string.")
    elif not os.path.exists(output_dir):
        raise OSError("Extract_color_channel_means: output_dir is not a valid path.")
    elif not os.path.isdir(output_dir):
        raise OSError("Extract_color_channel_means: output_dir must be a path string to a directory.")
    
    if not isinstance(color_space, int):
        if not isinstance(color_space, str):
            raise TypeError("Extract_color_channel_means: color_space must be an int or str.")
    if isinstance(color_space, str):
        if str.lower(color_space) not in ["rgb", "hsv", "grayscale"]:
            raise ValueError("Extract_color_channel_means: unspecified color space.")
        else:
            if str.lower(color_space) == "rgb":
                color_space = COLOR_SPACE_RGB
            if str.lower(color_space) == "hsv":
                color_space = COLOR_SPACE_HSV
            if str.lower(color_space) == "grayscale":
                color_space = COLOR_SPACE_GRAYSCALE

    if isinstance(color_space, int):
        if color_space not in [COLOR_SPACE_RGB, COLOR_SPACE_HSV, COLOR_SPACE_GRAYSCALE]:
            raise ValueError("Extract_color_channel_means: unspecified color space.")
    
    if not isinstance(with_sub_dirs, bool):
        raise TypeError("Extract_color_channel_means: with_sub_dirs must be a boolean.")
    
    if not isinstance(min_detection_confidence, float):
        raise TypeError("Extract_color_channel_means: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        raise ValueError("Extract_color_channel_means: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        raise TypeError("Extract_color_channel_means: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        raise ValueError("Extract_color_channel_means: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Defining mediapipe facemesh task
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
    files_to_process = []
    capture = None
    csv = None

    # Creating a list of file path strings
    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    # Create an output directory for the csv files
    if not os.path.isdir(output_dir + "\\Color_Channel_Means"):
        os.mkdir(output_dir + "\\Color_Channel_Means")
    output_dir = output_dir + "\\Color_Channel_Means"
    
    for file in files_to_process:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        capture = cv.VideoCapture(file)
        if not capture.isOpened():
            print("Extract_color_channel_means: Error opening videoCapture object.")
            sys.exit(1)
        
        # Writing the column headers to csv
        if color_space == COLOR_SPACE_RGB:
            csv = open(output_dir + "\\" + filename + "_RGB.csv", "x")
            csv.write("Timestamp,Red,Green,Blue\n")
        elif color_space == COLOR_SPACE_HSV:
            csv = open(output_dir + "\\" + filename + "_HSV.csv", "x")
            csv.write("Timestamp,Hue,Saturation,Value\n")
        elif color_space == COLOR_SPACE_GRAYSCALE:
            csv = open(output_dir + "\\" + filename + "_GRAYSCALE.csv", "x")
            csv.write("Timestamp,Value\n")
    
    while True:
        success, frame = capture.read()
        if not success:
            break

        face_mesh_results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        landmark_screen_coords = []

        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:

                # Convert normalised landmark coordinates to x-y pixel coordinates
                for id,lm in enumerate(face_landmarks.landmark):
                    ih, iw, ic = frame.shape
                    x,y = int(lm.x * iw), int(lm.y * ih)
                    landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
        else:
            continue
        
        if mask_face:
            face_outline_coords = []
            # Face oval screen coordinates
            for cur_source, cur_target in FACE_OVAL_TIGHT_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                face_outline_coords.append((source.get('x'),source.get('y')))
                face_outline_coords.append((target.get('x'),target.get('y')))
            
            # Use screen coordinates to create boolean mask
            oval_mask = np.zeros((frame.shape[0],frame.shape[1]))
            oval_mask = cv.fillConvexPoly(oval_mask, np.array(face_outline_coords), 1)
            oval_mask = oval_mask.astype(bool)

            bin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            bin_mask[oval_mask] = 255

            if color_space == COLOR_SPACE_RGB:
                blue, green, red, *_ = cv.mean(frame, bin_mask)
                timestamp = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                csv.write(f"{timestamp:.5f},{red:.5f},{green:.5f},{blue:.5f}\n")

            elif color_space == COLOR_SPACE_HSV:
                hue, sat, val, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_mask)
                timestamp = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                csv.write(f"{timestamp:.5f},{hue:.5f},{sat:.5f},{val:.5f}\n")
            
            elif color_space == COLOR_SPACE_GRAYSCALE:
                val, *_ = cv.mean(cv.cvtColor(frame, color_space), bin_mask)
                timestamp = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                csv.write(f"{timestamp:.5f},{val:.5f}\n")

        else:
            if color_space == COLOR_SPACE_RGB:
                blue, green, red, *_ = cv.mean(frame)
                timestamp = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                csv.write(f"{timestamp:.5f},{red:.5f},{green:.5f},{blue:.5f}\n")

            elif color_space == COLOR_SPACE_HSV:
                hue, sat, val, *_ = cv.mean(cv.cvtColor(frame, color_space))
                timestamp = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                csv.write(f"{timestamp:.5f},{hue:.5f},{sat:.5f},{val:.5f}\n")
            
            elif color_space == COLOR_SPACE_GRAYSCALE:
                val, *_ = cv.mean(cv.cvtColor(frame, color_space))
                timestamp = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                csv.write(f"{timestamp:.5f},{val:.5f}\n")
    
    capture.release()
    csv.close()

def face_color_shift(input_dir:str, output_dir:str, onset_t:float = 0.0, offset_t:float = 0.0, shift_magnitude: float = 8.0, timing_func:Callable[...,float] = sigmoid, 
                     shift_color:str|int = COLOR_RED, with_sub_dirs:bool = False, min_detection_confidence:float = 0.5, min_tracking_confidence:float = 0.5) -> None: 
    """For each image or video file contained in input_dir, the function applies a weighted color shift to the face region, 
    outputting each resulting file in output_dir. Weights are calculated using a passed timing function, that returns
    a float in the normalised range [0,1].
    (NOTE there is currently no checking to ensure timing function outputs are normalised)

    Parameters
    ----------

    input_dir: str
        A path string to the directory containing input video files.

    output_dir: str
        A path string to the directory where outputted video files will be saved.
    
    onset_t: float
        The onset time of the colour shifting.
    
    offset_t: float
        The offset time of the colour shifting.
    
    shift_magnitude: float
        The maximum units to shift the colour temperature by, during peak onset.
    
    timingFunc: Function() -> float
        Any function that takes at least one input float (time), and returns a float.

    shift_color: str, int
        Either a string literal specifying the color of choice, or a predefined integer constant.
    
    with_sub_dirs: bool
        A boolean flag indicating whether the input directory contains nested directories.
    
    min_detection_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.

    min_tracking_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
        
    Raises
    ------
    
    TypeError
        Given invalid parameter types.
    OSError
        Given invalid directory paths.
    ValueError:
        If provided timing_func does not return a normalised float value.
    """

    global FACE_OVAL_PATH
    global RIGHT_IRIS_PATH
    global LEFT_IRIS_PATH
    global LIPS_TIGHT_PATH
    global COLOR_RED
    global COLOR_BLUE
    global COLOR_GREEN
    global COLOR_YELLOW
    singleFile = False
    static_image_mode = False

    def shift_color_temp(img: cv2.typing.MatLike, img_mask: cv2.typing.MatLike | None, shift_weight: float, max_color_shift: float = 8.0, 
                    shift_color: str|int = COLOR_RED) -> cv2.typing.MatLike:
        """Takes in an image and a mask of the same shape, and shifts the specified color temperature by (weight * max_shift) 
        units in the masked region of the image. This function makes use of the CIE La*b* perceptually uniform color space to 
        perform natural looking color shifts on the face.

        Parameters
        ----------

        img: Matlike
            An input still image or video frame.

        img_mask: Matlike
            A binary image with the same shape as img.

        shift_weight: float
            The current shifting weight; a float in the range [0,1] returned from a timing function. 

        max_color_shift: float
            The maximum units to shift a* (red-green) or b* (blue-yellow) of the Lab* color space.
        
        shift_color: str, int
            An integer or string literal specifying which color will be applied to the input image.
                
        Raises
        ------

        TypeError
            On invalid input parameter types.
        ValueError 
            If an undefined color value is passed, or non-matching image and mask shapes are provided.

        Returns
        -------

        result: Matlike
            The input image, color-shifted in the region specified by the input mask. 
        """

        global COLOR_RED
        global COLOR_BLUE
        global COLOR_GREEN
        global COLOR_YELLOW

        # Convert input image to CIE La*b* color space (perceptually uniform space)
        img_LAB = cv.cvtColor(img, cv.COLOR_BGR2LAB).astype(np.float32)
        l,a,b = cv.split(img_LAB)

        if shift_color == COLOR_RED or str.lower(shift_color) == "red":
            a = np.where(img_mask==255, a + (shift_weight * max_color_shift), a)
            np.clip(a, -128, 127)
        if shift_color == COLOR_BLUE or str.lower(shift_color) == "blue":
            b = np.where(img_mask==255, b - (shift_weight * max_color_shift), b)
            np.clip(a, -128, 127)
        if shift_color == COLOR_GREEN or str.lower(shift_color) == "green":
            a = np.where(img_mask==255, a - (shift_weight * max_color_shift), a)
            np.clip(a, -128, 127)
        if shift_color == COLOR_YELLOW or str.lower(shift_color) == "yellow":
            b = np.where(img_mask==255, b + (shift_weight * max_color_shift), b)
            np.clip(a, -128, 127)
        
        img_LAB = cv.merge([l,a,b])
        
        # Convert CIE La*b* back to BGR
        result = cv.cvtColor(img_LAB.astype(np.uint8), cv.COLOR_LAB2BGR)
        return result
    
    # Performing checks on function parameters
    if not isinstance(input_dir, str):
        raise TypeError("Face_color_shift: invalid type for parameter input_dir.")
    elif not os.path.exists(input_dir):
        raise OSError("Face_color_shift: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(input_dir):
        singleFile = True
    
    if not isinstance(output_dir, str):
        raise TypeError("Face_color_shift: parameter output_dir must be a str.")
    elif not os.path.exists(output_dir):
        raise OSError("Face_color_shift: output directory path is not a valid path, or the directory does not exist.")
    elif not os.path.isdir(output_dir):
        raise ValueError("Face_color_shift: output_dir must be a valid path to a directory.")
    
    if not isinstance(onset_t, float):
        raise TypeError("Face_color_shift: parameter onset_t must be a float.")
    if not isinstance(offset_t, float):
        raise TypeError("Face_color_shift: parameter offset_t must be a float.")
    if not isinstance(shift_magnitude, float):
        raise TypeError("Face_color_shift: parameter shift_magnitude must be a float.")
    
    if isinstance(timing_func, Callable):
        if not isinstance(timing_func(1.0), float):
            raise ValueError("Face_color_shift: timing_func must return a float value.")
    # add check for return value in range [0,1]

    if isinstance(shift_color, str):
        if str.lower(shift_color) not in ["red", "green", "blue", "yellow"]:
            raise ValueError("Face_color_shift: shift_color must be one of: red, green, blue, yellow.")
    elif isinstance(shift_color, int):
        if shift_color not in [COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW]:
            raise ValueError("Face_color_shift: shift_color must be one of: red, green, blue, yellow.")
    else:
        raise TypeError("Face_color_shift: shift_color must be of type str or int.")

    if not isinstance(with_sub_dirs, bool):
        raise TypeError("Face_color_shift: parameter with_sub_dirs must be of type bool.")
    
    if not isinstance(min_detection_confidence, float):
        raise TypeError("Face_color_shift: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        raise ValueError("Face_color_shift: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        raise TypeError("Face_color_shift: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        raise ValueError("Face_color_shift: parameter min_tracking_confidence must be in the range [0,1].")

    # Creating a list of file path strings to iterate through when processing
    files_to_process = []

    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    # Creating named output directories for video output
    if not os.path.isdir(output_dir + "\\Color_Shifted"):
        os.mkdir(output_dir + "\\Color_Shifted")
    output_dir = output_dir + "\\Color_Shifted"
    
    for file in files_to_process:
            
        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        capture = None
        result = None
        cap_duration = None

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".jpg" | ".jpeg" | ".png" | ".bmp":
                static_image_mode = True
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = True, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case _:
                print("Face_color_shift: Incompatible video or image file type. Please see utils.transcode_video_to_mp4().")
                sys.exit(1)
        
        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                print("Face_color_shift: Error opening video file.")
                sys.exit(1)
            
            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(output_dir + "\\" + filename + "_color_shifted" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                print("Face_color_shift: Error opening VideoWriter object.")
                sys.exit(1)
            
            # Getting the video duration for weight calculations
            frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv.CAP_PROP_FPS)
            cap_duration = float(frame_count)/float(fps)

            if offset_t == 0.0:
                offset_t = cap_duration // 1
            
        while True:
            frame = None
            if static_image_mode:
                frame = cv.imread(file)
            else:
                success, frame = capture.read()
                if not success:
                    break    

            face_mesh_results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            landmark_screen_coords = []

            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:

                    # Convert normalised landmark coordinates to x-y pixel coordinates
                    for id,lm in enumerate(face_landmarks.landmark):
                        ih, iw, ic = frame.shape
                        x,y = int(lm.x * iw), int(lm.y * ih)
                        landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
            else:
                continue
            
            # Getting screen coordinates of facial landmarks
            le_screen_coords = []
            re_screen_coords = []
            lips_screen_coords = []
            face_outline_coords = []

            # Left eye screen coordinates
            for cur_source, cur_target in LEFT_IRIS_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                le_screen_coords.append((source.get('x'),source.get('y')))
                le_screen_coords.append((target.get('x'),target.get('y')))
            
            # Right eye screen coordinates
            for cur_source, cur_target in RIGHT_IRIS_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                re_screen_coords.append((source.get('x'),source.get('y')))
                re_screen_coords.append((target.get('x'),target.get('y')))

            # Lips screen coordinates
            for cur_source, cur_target in LIPS_TIGHT_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                lips_screen_coords.append((source.get('x'),source.get('y')))
                lips_screen_coords.append((target.get('x'),target.get('y')))
            
            # Face oval screen coordinates
            for cur_source, cur_target in FACE_OVAL_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                face_outline_coords.append((source.get('x'),source.get('y')))
                face_outline_coords.append((target.get('x'),target.get('y')))

            # Creating boolean masks for the facial landmarks 
            le_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
            le_mask = le_mask.astype(bool)

            re_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
            re_mask = re_mask.astype(bool)

            lip_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            lip_mask = cv.fillConvexPoly(lip_mask, np.array(lips_screen_coords), 1)
            lip_mask = lip_mask.astype(bool)

            oval_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            oval_mask = cv.fillConvexPoly(oval_mask, np.array(face_outline_coords), 1)
            oval_mask = oval_mask.astype(bool)

            # Masking the face oval
            masked_frame = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            masked_frame[oval_mask] = 255
            masked_frame[le_mask] = 0
            masked_frame[re_mask] = 0
            masked_frame[lip_mask] = 0
            
            # Otsu thresholding to seperate foreground and background
            grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grey_blurred = cv.GaussianBlur(grey_frame, (7,7), 0)
            thresh_val, thresholded = cv.threshold(grey_blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

            # Adding a temporary image border to allow for correct floodfill behaviour
            bordered_thresholded = cv.copyMakeBorder(thresholded, 10, 10, 10, 10, cv.BORDER_CONSTANT)
            floodfilled = bordered_thresholded.copy()
            cv.floodFill(floodfilled, None, (0,0), 255)

            # Removing temporary border and creating foreground mask
            floodfilled = floodfilled[10:-10, 10:-10]
            floodfilled = cv.bitwise_not(floodfilled)
            foreground = cv.bitwise_or(thresholded, floodfilled)
            
            if not static_image_mode:
                # Getting the current video timestamp
                dt = capture.get(cv.CAP_PROP_POS_MSEC)/1000

                if dt < onset_t:
                    result.write(frame)
                elif dt < offset_t:
                    weight = timing_func(dt)
                    frame_coloured = shift_color_temp(img=frame, img_mask=masked_frame, shift_weight=weight, shift_color=shift_color, max_color_shift=shift_magnitude)
                    frame_coloured[foreground == 0] = frame[foreground == 0]
                    result.write(frame_coloured)
                else:
                    dt = cap_duration - dt
                    weight = timing_func(dt)
                    frame_coloured = shift_color_temp(img=frame, img_mask=masked_frame, shift_weight=weight, shift_color=shift_color, max_color_shift=shift_magnitude)
                    frame_coloured[foreground == 0] = frame[foreground == 0]
                    result.write(frame_coloured)
            
            else:
                frame_coloured = shift_color_temp(img=frame, img_mask=masked_frame, shift_weight=1.0, shift_color=shift_color, max_color_shift=shift_magnitude)
                frame_coloured[foreground == 0] = frame[foreground == 0]
                success = cv.imwrite(output_dir + "\\" + filename + "_color_shifted" + extension, frame_coloured)

                if not success:
                    print("Face_color_shift: cv2.imwrite error.")
                    sys.exit(1)
                break

        if not static_image_mode:
            capture.release()
            result.release()

def face_saturation_shift(input_dir:str, output_dir:str, onset_t:float = 0.0, offset_t:float = 0.0, shift_magnitude:float = -8.0, 
                          timing_func:Callable[..., float] = sigmoid, with_sub_dirs:bool = False, min_detection_confidence:float = 0.5,
                           min_tracking_confidence:float = 0.5) -> None:
    """For each image or video file contained in input_dir, the function applies a weighted saturation shift to the face region, 
    outputting each processed file to output_dir. Weights are calculated using a passed timing function, that returns
    a float in the normalised range [0,1].
    (NOTE there is currently no checking to ensure timing function outputs are normalised)

    Parameters
    ----------

    input_dir: str
        A path string to the directory containing input video files.

    output_dir: str
        A path string to the directory where outputted video files will be saved.
    
    onset_t: float
        The onset time of the colour shifting.
    
    offset_t: float
        The offset time of the colour shifting.
    
    shift_magnitude: float
        The maximum units to shift the saturation by, during peak onset.
    
    timingFunc: Function() -> float
        Any function that takes at least one input float (time), and returns a float.
    
    with_sub_dirs: bool
        A boolean flag indicating whether the input directory contains nested directories.
    
    min_detection_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.

    min_tracking_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
        
    Raises
    ------
    
    TypeError
        Given invalid parameter types.
    OSError
        Given invalid directory paths.
    ValueError:
        If provided timing_func does not return a normalised float value.
    """

    global FACE_OVAL_PATH
    global RIGHT_IRIS_PATH
    global LEFT_IRIS_PATH
    global LIPS_TIGHT_PATH
    singleFile = False
    static_image_mode = False
    
    # Performing checks on function parameters
    if not isinstance(input_dir, str):
        raise TypeError("Face_saturation_shift: invalid type for parameter input_dir.")
    elif not os.path.exists(input_dir):
        raise OSError("Face_saturation_shift: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(input_dir):
        singleFile = True
    
    if not isinstance(output_dir, str):
        raise TypeError("Face_saturation_shift: parameter output_dir must be a str.")
    elif not os.path.exists(output_dir):
        raise OSError("Face_saturation_shift: output directory path is not a valid path, or the directory does not exist.")
    elif not os.path.isdir(output_dir):
        raise ValueError("Face_saturation_shift: output_dir must be a valid path to a directory.")
    
    if not isinstance(onset_t, float):
        raise TypeError("Face_saturation_shift: parameter onset_t must be a float.")
    if not isinstance(offset_t, float):
        raise TypeError("Face_saturation_shift: parameter offset_t must be a float.")
    if not isinstance(shift_magnitude, float):
        raise TypeError("Face_saturation_shift: parameter shift_magnitude must be a float.")
    
    if isinstance(timing_func, Callable):
        if not isinstance(timing_func(1.0), float):
            raise ValueError("Face_saturation_shift: timing_func must return a float value.")
    # add check for return value in range [0,1]

    if not isinstance(with_sub_dirs, bool):
        raise TypeError("Face_saturation_shift: parameter with_sub_dirs must be of type bool.")
    
    if not isinstance(min_detection_confidence, float):
        raise TypeError("Face_saturation_shift: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        raise ValueError("Face_saturation_shift: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        raise TypeError("Face_saturation_shift: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        raise ValueError("Face_saturation_shift: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = []

    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    # Creating named output directories for video output
    if not os.path.isdir(output_dir + "\\Sat_Shifted"):
        os.mkdir(output_dir + "\\Sat_Shifted")
    output_dir = output_dir + "\\Sat_Shifted"
    
    for file in files_to_process:
            
        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        capture = None
        result = None
        cap_duration = None

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".jpg" | ".jpeg" | ".png" | ".bmp":
                static_image_mode = True
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = True, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case _:
                print("Face_saturation_shift: Incompatible video or image file type. Please see psyfaceutils.transcode_video_to_mp4().")
                sys.exit(1)
        
        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                print("Face_saturation_shift: Error opening video file.")
                sys.exit(1)
            
            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(output_dir + "\\" + filename + "_sat_shifted" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                print("Face_saturation_shift: Error opening VideoWriter object.")
                sys.exit(1)
            
            # Getting the video duration for weight calculations
            frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv.CAP_PROP_FPS)
            cap_duration = float(frame_count)/float(fps)

            if offset_t == 0.0:
                offset_t = cap_duration // 1
            
        while True:
            frame = None
            if static_image_mode:
                frame = cv.imread(file)
            else:
                success, frame = capture.read()
                if not success:
                    break    

            face_mesh_results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            landmark_screen_coords = []

            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:

                    # Convert normalised landmark coordinates to x-y pixel coordinates
                    for id,lm in enumerate(face_landmarks.landmark):
                        ih, iw, ic = frame.shape
                        x,y = int(lm.x * iw), int(lm.y * ih)
                        landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
            else:
                continue
            
            # Getting screen coordinates of facial landmarks
            le_screen_coords = []
            re_screen_coords = []
            lips_screen_coords = []
            face_outline_coords = []

            # Left eye screen coordinates
            for cur_source, cur_target in LEFT_IRIS_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                le_screen_coords.append((source.get('x'),source.get('y')))
                le_screen_coords.append((target.get('x'),target.get('y')))
            
            # Right eye screen coordinates
            for cur_source, cur_target in RIGHT_IRIS_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                re_screen_coords.append((source.get('x'),source.get('y')))
                re_screen_coords.append((target.get('x'),target.get('y')))

            # Lips screen coordinates
            for cur_source, cur_target in LIPS_TIGHT_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                lips_screen_coords.append((source.get('x'),source.get('y')))
                lips_screen_coords.append((target.get('x'),target.get('y')))
            
            # Face oval screen coordinates
            for cur_source, cur_target in FACE_OVAL_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                face_outline_coords.append((source.get('x'),source.get('y')))
                face_outline_coords.append((target.get('x'),target.get('y')))

            # Creating boolean masks for the facial landmarks 
            le_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
            le_mask = le_mask.astype(bool)

            re_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
            re_mask = re_mask.astype(bool)

            lip_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            lip_mask = cv.fillConvexPoly(lip_mask, np.array(lips_screen_coords), 1)
            lip_mask = lip_mask.astype(bool)

            oval_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            oval_mask = cv.fillConvexPoly(oval_mask, np.array(face_outline_coords), 1)
            oval_mask = oval_mask.astype(bool)

            # Masking the face oval
            masked_frame = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            masked_frame[oval_mask] = 255
            masked_frame[le_mask] = 0
            masked_frame[re_mask] = 0
            masked_frame[lip_mask] = 0
            
            # Otsu thresholding to seperate foreground and background
            grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grey_blurred = cv.GaussianBlur(grey_frame, (7,7), 0)
            thresh_val, thresholded = cv.threshold(grey_blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

            # Adding a temporary image border to allow for correct floodfill behaviour
            bordered_thresholded = cv.copyMakeBorder(thresholded, 10, 10, 10, 10, cv.BORDER_CONSTANT)
            floodfilled = bordered_thresholded.copy()
            cv.floodFill(floodfilled, None, (0,0), 255)

            # Removing temporary border and creating foreground mask
            floodfilled = floodfilled[10:-10, 10:-10]
            floodfilled = cv.bitwise_not(floodfilled)
            foreground = cv.bitwise_or(thresholded, floodfilled)
            
            if not static_image_mode:
                # Getting the current video timestamp
                dt = capture.get(cv.CAP_PROP_POS_MSEC)/1000

                if dt < onset_t:
                    result.write(frame)
                elif dt < offset_t:
                    shift_weight = timing_func(dt)
                    img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV).astype(np.float32)

                    h,s,v = cv.split(img_hsv)
                    s = np.where(masked_frame == 255, s + (shift_weight * shift_magnitude), s)
                    np.clip(s,0,255)
                    img_hsv = cv.merge([h,s,v])

                    img_bgr = cv.cvtColor(img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
                    img_bgr[foreground == 0] = frame[foreground == 0]
                    result.write(img_bgr)
                else:
                    dt = cap_duration - dt
                    shift_weight = timing_func(dt)
                    img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV).astype(np.float32)

                    h,s,v = cv.split(img_hsv)
                    s = np.where(masked_frame == 255, s + (shift_weight * shift_magnitude), s)
                    np.clip(s,0,255)
                    img_hsv = cv.merge([h,s,v])

                    img_bgr = cv.cvtColor(img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
                    img_bgr[foreground == 0] = frame[foreground == 0]
                    result.write(img_bgr)

            else:
                img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV).astype(np.float32)

                h,s,v = cv.split(img_hsv)
                s = np.where(masked_frame == 255, s + (1.0 * shift_magnitude), s)
                np.clip(s,0,255)
                img_hsv = cv.merge([h,s,v])

                img_bgr = cv.cvtColor(img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
                img_bgr[foreground == 0] = frame[foreground == 0]
                success = cv.imwrite(output_dir + "\\" + filename + "_sat_shifted" + extension, img_bgr)

                if not success:
                    print("Face_saturation_shift: cv2.imwrite error.")
                    sys.exit(1)

                break

        if not static_image_mode:
            capture.release()
            result.release()

def face_brightness_shift(input_dir:str, output_dir:str, onset_t:float = 0.0, offset_t:float = 0.0, shift_magnitude:int = 20, 
                        timing_func:Callable[..., float] = sigmoid, with_sub_dirs:bool = False, min_detection_confidence:float = 0.5,
                        min_tracking_confidence:float = 0.5) -> None:
    """For each image or video file contained in input_dir, the function applies a weighted brightness shift to the face region, 
    outputting each processed file to output_dir. Weights are calculated using a passed timing function, that returns a float in the normalised range [0,1].
    (NOTE there is currently no checking to ensure timing function outputs are normalised)

    Parameters
    ----------

    input_dir: str
        A path string to the directory containing input video files.

    output_dir: str
        A path string to the directory where outputted video files will be saved.
    
    onset_t: float
        The onset time of the brightness shifting.
    
    offset_t: float
        The offset time of the brightness shifting.
    
    shift_magnitude: float
        The maximum units to shift the brightness by, during peak onset.
    
    timingFunc: Function() -> float
        Any function that takes at least one input float (time), and returns a float.
    
    with_sub_dirs: bool
        A boolean flag indicating whether the input directory contains nested directories.
    
    min_detection_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.

    min_tracking_confidence: float
        A normalised float value in the range [0,1], this parameter is passed as a specifier to the mediapipe 
        FaceMesh constructor.
        
    Raises
    ------
    
    TypeError
        Given invalid parameter types.
    OSError
        Given invalid directory paths.
    ValueError:
        If provided timing_func does not return a normalised float value.
    """

    global FACE_OVAL_PATH
    global RIGHT_IRIS_PATH
    global LEFT_IRIS_PATH
    global LIPS_TIGHT_PATH
    singleFile = False
    static_image_mode = False
    
    # Performing checks on function parameters
    if not isinstance(input_dir, str):
        raise TypeError("Face_brightness_shift: invalid type for parameter input_dir.")
    elif not os.path.exists(input_dir):
        raise OSError("Face_brightness_shift: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(input_dir):
        singleFile = True
    
    if not isinstance(output_dir, str):
        raise TypeError("Face_brightness_shift: parameter output_dir must be a str.")
    elif not os.path.exists(output_dir):
        raise OSError("Face_brightness_shift: output directory path is not a valid path, or the directory does not exist.")
    elif not os.path.isdir(output_dir):
        raise ValueError("Face_brightness_shift: output_dir must be a valid path to a directory.")
    
    if not isinstance(onset_t, float):
        raise TypeError("Face_brightness_shift: parameter onset_t must be a float.")
    if not isinstance(offset_t, float):
        raise TypeError("Face_brightness_shift: parameter offset_t must be a float.")
    if not isinstance(shift_magnitude, int):
        raise TypeError("Face_brightness_shift: parameter shift_magnitude must be an int.")
    
    if isinstance(timing_func, Callable):
        if not isinstance(timing_func(1.0), float):
            raise ValueError("Face_brightness_shift: timing_func must return a float value.")

    if not isinstance(with_sub_dirs, bool):
        raise TypeError("Face_brightness_shift: parameter with_sub_dirs must be of type bool.")
    
    if not isinstance(min_detection_confidence, float):
        raise TypeError("Face_brightness_shift: parameter min_detection_confidence must be of type float.")
    elif min_detection_confidence < 0 or min_detection_confidence > 1:
        raise ValueError("Face_brightness_shift: parameter min_detection_confidence must be in the range [0,1].")
    
    if not isinstance(min_tracking_confidence, float):
        raise TypeError("Face_brightness_shift: parameter min_tracking_confidence must be of type float.")
    elif min_tracking_confidence < 0 or min_tracking_confidence > 1:
        raise ValueError("Face_brightness_shift: parameter min_tracking_confidence must be in the range [0,1].")
    
    # Creating a list of file path strings to iterate through when processing
    files_to_process = []

    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
        files_to_process = [input_dir + "\\" + file for file in os.listdir(input_dir)]
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    # Creating named output directories for video output
    if not os.path.isdir(output_dir + "\\Brightness_Shifted"):
        os.mkdir(output_dir + "\\Brightness_Shifted")
    output_dir = output_dir + "\\Brightness_Shifted"
    
    for file in files_to_process:
            
        # Filetype is used to determine the functions running mode
        filename, extension = os.path.splitext(os.path.basename(file))
        codec = None
        capture = None
        result = None
        cap_duration = None

        # Using the file extension to sniff video codec or image container for images
        match extension:
            case ".mp4":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".mov":
                codec = "MP4V"
                static_image_mode = False
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case ".jpg" | ".jpeg" | ".png" | ".bmp":
                static_image_mode = True
                face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = True, 
                            min_detection_confidence = min_detection_confidence, min_tracking_confidence = min_tracking_confidence)
            case _:
                print("Face_brightness_shift: Incompatible video or image file type. Please see psyfaceutils.transcode_video_to_mp4().")
                sys.exit(1)
        
        if not static_image_mode:
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                print("Face_brightness_shift: Error opening video file.")
                sys.exit(1)
            
            size = (int(capture.get(3)), int(capture.get(4)))

            result = cv.VideoWriter(output_dir + "\\" + filename + "_light_shifted" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                print("Face_brightness_shift: Error opening VideoWriter object.")
                sys.exit(1)
            
            # Getting the video duration for weight calculations
            frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv.CAP_PROP_FPS)
            cap_duration = float(frame_count)/float(fps)

            if offset_t == 0.0:
                offset_t = cap_duration // 1
            
        while True:
            frame = None
            if static_image_mode:
                frame = cv.imread(file)
            else:
                success, frame = capture.read()
                if not success:
                    break    

            face_mesh_results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            landmark_screen_coords = []

            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:

                    # Convert normalised landmark coordinates to x-y pixel coordinates
                    for id,lm in enumerate(face_landmarks.landmark):
                        ih, iw, ic = frame.shape
                        x,y = int(lm.x * iw), int(lm.y * ih)
                        landmark_screen_coords.append({'id':id, 'x':x, 'y':y})
            else:
                continue
            
            # Getting screen coordinates of facial landmarks
            le_screen_coords = []
            re_screen_coords = []
            lips_screen_coords = []
            face_outline_coords = []

            # Left eye screen coordinates
            for cur_source, cur_target in LEFT_IRIS_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                le_screen_coords.append((source.get('x'),source.get('y')))
                le_screen_coords.append((target.get('x'),target.get('y')))
            
            # Right eye screen coordinates
            for cur_source, cur_target in RIGHT_IRIS_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                re_screen_coords.append((source.get('x'),source.get('y')))
                re_screen_coords.append((target.get('x'),target.get('y')))

            # Lips screen coordinates
            for cur_source, cur_target in LIPS_TIGHT_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                lips_screen_coords.append((source.get('x'),source.get('y')))
                lips_screen_coords.append((target.get('x'),target.get('y')))
            
            # Face oval screen coordinates
            for cur_source, cur_target in FACE_OVAL_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                face_outline_coords.append((source.get('x'),source.get('y')))
                face_outline_coords.append((target.get('x'),target.get('y')))

            # Creating boolean masks for the facial landmarks 
            le_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            le_mask = cv.fillConvexPoly(le_mask, np.array(le_screen_coords), 1)
            le_mask = le_mask.astype(bool)

            re_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            re_mask = cv.fillConvexPoly(re_mask, np.array(re_screen_coords), 1)
            re_mask = re_mask.astype(bool)

            lip_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            lip_mask = cv.fillConvexPoly(lip_mask, np.array(lips_screen_coords), 1)
            lip_mask = lip_mask.astype(bool)

            oval_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            oval_mask = cv.fillConvexPoly(oval_mask, np.array(face_outline_coords), 1)
            oval_mask = oval_mask.astype(bool)

            # Masking the face oval
            masked_frame = np.zeros((frame.shape[0],frame.shape[1],1), dtype=np.uint8)
            masked_frame[oval_mask] = 255
            masked_frame[le_mask] = 0
            masked_frame[re_mask] = 0
            masked_frame[lip_mask] = 0
            
            # Otsu thresholding to seperate foreground and background
            grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grey_blurred = cv.GaussianBlur(grey_frame, (7,7), 0)
            thresh_val, thresholded = cv.threshold(grey_blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

            # Adding a temporary image border to allow for correct floodfill behaviour
            bordered_thresholded = cv.copyMakeBorder(thresholded, 10, 10, 10, 10, cv.BORDER_CONSTANT)
            floodfilled = bordered_thresholded.copy()
            cv.floodFill(floodfilled, None, (0,0), 255)

            # Removing temporary border and creating foreground mask
            floodfilled = floodfilled[10:-10, 10:-10]
            floodfilled = cv.bitwise_not(floodfilled)
            foreground = cv.bitwise_or(thresholded, floodfilled)

            if not static_image_mode:
                # Getting the current video timestamp
                dt = capture.get(cv.CAP_PROP_POS_MSEC)/1000

                if dt < onset_t:
                    result.write(frame)
                elif dt < offset_t:
                    weight = timing_func(dt)

                    img_brightened = frame.copy()
                    img_brightened = np.where(masked_frame == 255, cv.convertScaleAbs(src=img_brightened, alpha=1, beta=(weight * shift_magnitude)), frame)

                    #cv.convertScaleAbs(img_brightened, img_brightened, 1, (weight * shift_magnitude))

                    img_brightened[foreground == 0] = frame[foreground == 0]
                    result.write(img_brightened)
                else:
                    dt = cap_duration - dt
                    weight = timing_func(dt)

                    img_brightened = frame.copy()
                    img_brightened = np.where(masked_frame == 255, cv.convertScaleAbs(src=img_brightened, alpha=1, beta=(weight * shift_magnitude)), frame)

                    #cv.convertScaleAbs(img_brightened, img_brightened, 1, (weight * shift_magnitude))

                    img_brightened[foreground == 0] = frame[foreground == 0]
                    result.write(img_brightened)

            else:
                # Brightening the image
                img_brightened = frame.copy()
                img_brightened = np.where(masked_frame == 255, cv.convertScaleAbs(src=img_brightened, alpha=1, beta=shift_magnitude), frame)

                # Making sure background remains unaffected
                img_brightened[foreground == 0] = frame[foreground == 0]

                success = cv.imwrite(output_dir + "\\" + filename + "_brightened" + extension, img_brightened)

                if not success:
                    print("Face_brightness_shift: cv2.imwrite error.")
                    sys.exit(1)

                break
        
        if not static_image_mode:
            capture.release()
            result.release()