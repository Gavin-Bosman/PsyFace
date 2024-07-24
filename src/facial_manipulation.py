import cv2 as cv
import cv2.typing
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import sys
from typing import Callable
from utils import *
#TODO face occlusion

def mask_face_region(input_dir:str, output_dir:str, mask_type:int = FACE_SKIN_ISOLATION, with_sub_dirs:bool = False) -> None:
    """Applies specified mask type to video files located in input_dir.

    Args:
        input_dir: String
            A path string of the directory containing videos to process.

        output_dir: String
            A path string of the directory where processed videos will be written to.

        mask_type: Integer
            An integer indicating the type of mask to apply to the input videos. This can be one of two options:
            either 1 for FACE_OVAL, or 2 for FACE_SKIN_ISOLATION.

        with_sub_dirs: Boolean, 
            Indicates if the input directory contains subfolders.
    
    Raises:
        ValueError: given an unknown mask type.
        TypeError: given invalid parameter types.
        OSError: given invalid path strings for in/output directories
    
    """

    global MASK_OPTIONS
    global FACE_OVAL
    global FACE_OVAL_TIGHT
    global FACE_SKIN_ISOLATION
    global LEFT_EYE_BROW_PATH
    global RIGHT_EYE_BROW_PATH
    global LIPS_PATH
    global FACE_OVAL_PATH
    global FACE_OVAL_TIGHT_PATH
    global COLOR_SPACE_RGB
    global COLOR_SPACE_HSV
    global COLOR_SPACE_GRAYSCALE
    global COLOR_SPACES
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                                                min_detection_confidence = 0.25, min_tracking_confidence = 0.75)
    singleFile = False

    # Type and value checks for function parameters
    if not isinstance(input_dir, str):
        raise TypeError("Mask_face_region: invalid type for parameter inputDirectory.")
    elif not os.path.exists(input_dir):
        raise OSError("Mask_face_region: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(input_dir):
        singleFile = True
    
    if not isinstance(output_dir, str):
        raise TypeError("Mask_face_region: invalid type for parameter outputDirectory.")
    elif not os.path.exists(output_dir):
        raise ValueError("Mask_face_region: output directory path is not a valid path, or the directory does not exist.")
    
    if mask_type not in MASK_OPTIONS:
        raise ValueError("Mask_face_region: maskType must be either 1: indicating FACE_OVAL, or 2: indicating FACE_SKIN_ISOLATION.")
    
    if not isinstance(with_sub_dirs, bool):
        raise TypeError("Mask_face_region: invalid type for parameter withSubDirectories.")

    # Creating a list of file names to iterate through when processing
    files_to_process = []
    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
         files_to_process = os.listdir(input_dir)
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    # Creating named output directories for video output
    if not os.path.isdir(output_dir + "\\Masked_Videos"):
        os.mkdir(output_dir + "\\Masked_Videos")
    output_dir = output_dir + "\\Masked_Videos"

    if mask_type == FACE_SKIN_ISOLATION:

        for file in files_to_process:

            # Initialize capture and writer objects
            filename, extension = os.path.splitext(os.path.basename(file))
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                print("Mask_face_region: Error opening VideoCapture object.")
                sys.exit(1)
                 
            codec = None

            match extension:
                case ".mp4":
                    codec = "H264"
                case ".mov":
                    codec = "H264"
                case _:
                    print("Mask_face_region: Incompatible video file type. Please see utils.transcode_video_to_mp4().")
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

                le_screen_coords = []
                re_screen_coords = []
                lips_screen_coords = []
                face_outline_coords = []

                # left eye screen coordinates
                for cur_source, cur_target in LEFT_EYE_BROW_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    le_screen_coords.append((source.get('x'),source.get('y')))
                    le_screen_coords.append((target.get('x'),target.get('y')))
                
                # right eye screen coordinates
                for cur_source, cur_target in RIGHT_EYE_BROW_PATH:
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
                for cur_source, cur_target in FACE_OVAL_TIGHT_PATH:
                    source = landmark_screen_coords[cur_source]
                    target = landmark_screen_coords[cur_target]
                    face_outline_coords.append((source.get('x'),source.get('y')))
                    face_outline_coords.append((target.get('x'),target.get('y')))

                # Creating the masked regions 
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
                
                # Masking out eye and mouth regions
                masked_frame = frame
                masked_frame[le_mask] = 0
                masked_frame[re_mask] = 0
                masked_frame[lip_mask] = 0

                # Last step, masking out the bounding face shape
                face_skin = np.zeros_like(masked_frame, dtype=np.uint8)
                face_skin[oval_mask] = masked_frame[oval_mask] 

                # Removing any face mesh artifacts
                grey = cv.cvtColor(face_skin, cv.COLOR_BGR2GRAY)
                white_mask = cv.inRange(grey, 220, 255)
                face_skin[white_mask == 255] = 0

                result.write(face_skin)
        
            capture.release()
            result.release()
    
    elif mask_type == FACE_OVAL:

        for file in files_to_process:

            # Initializing capture and writer objects
            filename, extension = os.path.splitext(file)
            capture = cv.VideoCapture(file)
            if not capture.isOpened:
                print("Mask_face_region: Error opening videoCapture object.")
                return -1

            codec = None

            match extension:
                case ".mp4":
                    codec = "H264"
                case ".mov":
                    codec = "H264"
                case _:
                    print("Mask_face_region: Incompatible video file type. Please see utils.transcode_video_to_mp4().")
                    sys.exit(1)
            
            size = (int(capture.get(3)), int(capture.get(4)))
            result = cv.VideoWriter(output_dir + "\\" + filename + "_masked" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                print("Mask_face_region: Error opening VideoWriter object.")
                return -1
            
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

                # Last step, masking out the bounding face shape
                face_skin = np.zeros_like(frame)
                face_skin[oval_mask] = frame[oval_mask] 

                # Removing any face mesh artifacts
                grey = cv.cvtColor(face_skin, cv.COLOR_BGR2GRAY)
                white_mask = cv.inRange(grey, 220, 255)
                face_skin[white_mask == 255] = 0

                result.write(face_skin)
        
            capture.release()
            result.release()
    
    elif mask_type == FACE_OVAL_TIGHT:

        for file in files_to_process:

            # Initializing capture and writer objects
            filename, extension = os.path.splitext(file)
            capture = cv.VideoCapture(file)
            if not capture.isOpened:
                print("Mask_face_region: Error opening videoCapture object.")
                return -1

            codec = None

            match extension:
                case ".mp4":
                    codec = "H264"
                case ".mov":
                    codec = "H264"
                case _:
                    print("Mask_face_region: Incompatible video file type. Please see utils.transcode_video_to_mp4().")
                    sys.exit(1)
            
            size = (int(capture.get(3)), int(capture.get(4)))
            result = cv.VideoWriter(output_dir + "\\" + filename + "_masked" + extension,
                                    cv.VideoWriter.fourcc(*codec), 30, size)
            if not result.isOpened():
                print("Mask_face_region: Error opening VideoWriter object.")
                return -1
            
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

                # Last step, masking out the bounding face shape
                face_skin = np.zeros_like(frame)
                face_skin[oval_mask] = frame[oval_mask] 

                # Removing any face mesh artifacts
                grey = cv.cvtColor(face_skin, cv.COLOR_BGR2GRAY)
                white_mask = cv.inRange(grey, 220, 255)
                face_skin[white_mask == 255] = 0

                result.write(face_skin)
        
            capture.release()
            result.release()

def extract_color_channel_means(input_dir:str, output_dir:str, color_space: int|str = COLOR_SPACE_RGB, 
                                with_sub_dirs:bool = False, mask_face:bool = True) -> None:
    """Extracts and outputs mean values of each color channel from the specified color space. Creates a new directory 
    'CSV_Output', where a csv file will be written for each input video file provided.

        Args:
            input_dir: str
                A path string to a directory containing the video files to be processed.

            output_dir: str
                A path string to a directory where outputted csv files will be written to.
            
            color_space: int | str
                A specifier for which color space to operate in.
            
            with_sub_dirs: bool
                Indicates whether the input directory contains subfolders
            
            mask_face: bool
                Indicates whether to mask the face region prior to extracting color means.
        
        Raises:
            TypeError: given invalid parameter types.
            ValueError: given an unrecognized color space.
            OSError: input or output directories are invalid paths.
    """
    
    # Global declarations and init
    global COLOR_SPACE_RGB
    global COLOR_SPACE_HSV
    global COLOR_SPACE_GRAYSCALE
    global FACE_OVAL_TIGHT_PATH
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                                                min_detection_confidence = 0.25, min_tracking_confidence = 0.75)
    singleFile = False
    files_to_process = []
    capture = None
    csv = None

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

    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
         files_to_process = os.listdir(input_dir)
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    # create an output directory for the csv files
    if not os.path.isdir(output_dir + "\\Color_Channel_Means"):
        os.mkdir(output_dir + "\\Color_Channel_Means")
    output_dir = os.path.join(output_dir, "\\Color_Channel_Means")
    
    for file in files_to_process:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        capture = cv.VideoCapture(file)
        if not capture.isOpened():
            print("Extract_color_channel_means: Error opening videoCapture object.")
            return -1
        
        # Writing the column headers to csv
        if color_space == COLOR_SPACE_RGB:
            csv = open(output_dir + "\\" + filename + "_RGB.csv", "w")
            csv.write("Timestamp,Red,Green,Blue\n")
        elif color_space == COLOR_SPACE_HSV:
            csv = open(output_dir + "\\" + filename + "_HSV.csv", "w")
            csv.write("Timestamp,Hue,Saturation,Value\n")
        elif color_space == COLOR_SPACE_GRAYSCALE:
            csv = open(output_dir + "\\" + filename + "_GRAYSCALE.csv", "w")
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

def shift_color_temp(img: cv2.typing.MatLike, img_mask: cv2.typing.MatLike | None, shift_weight: float, max_color_shift: float = 8.0, 
                    max_sat_shift: float = 0.0, shift_color: str|int = COLOR_RED, sat_only: bool = False) -> cv2.typing.MatLike:
    """Takes in an image and a mask of the same shape, and shifts the specified color temperature by (weight*max_shift) units in 
        the masked region of the image. This function makes use of the CIE Lab* perceptually uniform color space to perform natural looking
        color shifts on the face.

        Args:
            img: Matlike
                An input still image or video frame.

            img_mask: Matlike
                A binary image with the same shape as img.

            shift_weight: float
                The current shifting weight; a float in the range [0,1] returned from a timing function. 

            max_color_shift: float
                The maximum units to shift a* (red-green) or b* (blue-yellow) of the Lab* color space.
            
            max_sat_shift: float
                The maximum units to shift the images saturation by.
            
            shift_color: str | int
                An integer or string literal specifying which color will be applied to the input image.

            sat_only: bool
                A boolean flag that indicates if only the saturation is being modified.
            
        Raises:
            TypeError: on invalid input parameter types.
            ValueError: on undefined color values, or unmatching image and mask shapes.
        
        Returns:
            result: Matlike
                The input image, color-shifted in the region specified by the input mask. 

    """

    global COLOR_RED
    global COLOR_BLUE
    global COLOR_GREEN
    global COLOR_YELLOW
    
    # Type checking input parameters
    if not isinstance(img, cv2.typing.MatLike):
        raise TypeError("Color_shift: parameter img must be a Matlike.")
    if not isinstance(img_mask, cv2.typing.MatLike):
        raise TypeError("Color_shift: parameter mask must be a Matlike.")
    if img.shape[:2] != img_mask.shape[:2]:
        raise ValueError("Color_shift: image and mask have different shapes.")
    
    if not isinstance(shift_weight, float):
        raise TypeError("Color_shift: parameter weight must be of type float.")
    if not isinstance(max_color_shift, float):
        raise TypeError("Color_shift: parameter maxShift must be of type float.")
    if not isinstance(max_sat_shift, float):
        raise TypeError("Color_shift: parameter sat_delta must be of type float.")
    
    if isinstance(shift_color, str):
        if str.lower(shift_color) not in ["red", "green", "blue", "yellow"]:
            raise ValueError("Color_shift: color must be one of: red, green, blue, yellow.")
    elif isinstance(shift_color, int):
        if shift_color not in [COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW]:
            raise ValueError("Color_shift: color must be one of: red, green, blue, yellow.")
    else:
        raise TypeError("Color_shift: parameter color must be of type str or int.")
    
    result = None

    if not sat_only:

        img_LAB = None

        if max_sat_shift != 0.0:
            img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)
            h,s,v = cv.split(img_hsv)
            s = np.where(img_mask == 255, s + (shift_weight * max_sat_shift), s)
            np.clip(s,0,255)
            img_hsv = cv.merge([h,s,v])

            img_LAB = cv.cvtColor(img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)

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
        
    elif max_sat_shift != 0.0:
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)
        h,s,v = cv.split(img_hsv)
        s = np.where(img_mask == 255, s + (shift_weight * max_sat_shift), s)
        np.clip(s,0,255)
        img_hsv = cv.merge([h,s,v])

        result = cv.cvtColor(img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)

    # Cleaning up any background artifacts from changing color spaces
    grey = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    white_mask = cv.inRange(grey, 220, 255)
    result[white_mask == 255] = 255

    return result

def face_color_shift(input_dir:str, output_dir:str, onset_t:float, offset_t:float, max_color_shift: float = 8.0, max_sat_shift: float = 0.0,
                     timing_func:Callable[...,float] = sigmoid, shift_color:str|int = COLOR_RED, with_sub_dirs:bool = False, sat_only:bool = False) -> None: 
    """For each video file contained in input_dir, the function applies a weighted color shift to the face region, outputting 
    each resulting video to output_dir. Weights are calculated using a passed timing function, that returns a float in the normalised
    range [0,1]. (NOTE there is currently no checking to ensure timing function outputs are normalised)

        Args:
            input_dir: String
                A path string to the directory containing input video files.

            output_dir: String
                A path string to the directory where outputted video files will be saved.
            
            onset_t: float
                The onset time of the colour shifting.
            
            offset_t: float
                The offset time of the colour shifting.
            
            max_color_shift: float
                The maximum units to shift the colour temperature by, during peak onset.
            
            max_sat_shift: float
                The maximum units to shift the images saturation by, during peak onset.
            
            timingFunc: Function() -> float
                Any function that takes at least one input float (time), and returns a float.

            shift_color: String | int
                Either a string literal specifying the color of choice, or a predefined integer constant.
            
            with_sub_dirs: bool
                A boolean flag indicating whether the input directory contains nested directories.
            
            sat_only: bool
                A boolean flag indicating if only the saturation of the input file will be shifted.
        
        Throws:
            TypeError: given invalid parameter types.
            OSError: given invalid directory paths.
            ValueError: if timing_func does not return a float value.
    """

    global FACE_OVAL_PATH
    global RIGHT_EYE_PATH
    global LEFT_EYE_PATH
    global LIPS_TIGHT_PATH
    global COLOR_RED
    global COLOR_BLUE
    global COLOR_GREEN
    global COLOR_YELLOW
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                                                min_detection_confidence = 0.25, min_tracking_confidence = 0.75)
    singleFile = False
    
    # Performing checks on function parameters
    if not isinstance(input_dir, str):
        raise TypeError("Face_color_shift: invalid type for parameter inputDirectory.")
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
    if not isinstance(max_color_shift, float):
        raise TypeError("Face_color_shift: parameter max_color_shift must be a float.")
    if not isinstance(max_sat_shift, float):
        raise TypeError("Face_color_shift: parameter sat_delta must be a float.")
    
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
    
    if not isinstance(sat_only, bool):
        raise TypeError("Face_color_shift: parameter sat_only must be of type bool.")

    # Creating a list of file names to iterate through when processing
    files_to_process = []

    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
         files_to_process = os.listdir(input_dir)
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    # Creating named output directories for video output
    if not os.path.isdir(output_dir + "\\Color_Shifted_Videos"):
        os.mkdir(output_dir + "\\Color_Shifted_Videos")

    output_dir = output_dir + "\\Color_Shifted_Videos"
    
    for file in files_to_process:
            
        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        capture = cv.VideoCapture(file)
        if not capture.isOpened():
            print("Face_color_shift: Error opening video file.")
            sys.exit(1)
        
        codec = None

        match extension:
            case ".mp4":
                codec = "H264"
            case ".mov":
                codec = "H264"
            case _:
                print("Face_color_shift: Incompatible video file type. Please see utils.transcode_video_to_mp4().")
                sys.exit(1)
        
        size = (int(capture.get(3)), int(capture.get(4)))
        result = cv.VideoWriter(output_dir + "\\" + filename + "_color_shifted" + extension,
                                cv.VideoWriter.fourcc(*codec), 30, size)
        if not result.isOpened():
            print("Face_color_shift: Error opening VideoWriter object.")
            return -1

        # Getting the video duration for weight calculations
        frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv.CAP_PROP_FPS)
        cap_duration = float(frame_count)/float(fps)
            
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
            
            # Getting screen coordinates of facial landmarks
            le_screen_coords = []
            re_screen_coords = []
            lips_screen_coords = []
            face_outline_coords = []

            # Left eye screen coordinates
            for cur_source, cur_target in LEFT_EYE_PATH:
                source = landmark_screen_coords[cur_source]
                target = landmark_screen_coords[cur_target]
                le_screen_coords.append((source.get('x'),source.get('y')))
                le_screen_coords.append((target.get('x'),target.get('y')))
            
            # Right eye screen coordinates
            for cur_source, cur_target in RIGHT_EYE_PATH:
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

            # Isolating overall face skin for colouring
            face_mask = np.zeros((frame.shape[0],frame.shape[1]), dtype=np.uint8)
            face_mask[oval_mask] = 255

            # Eroding the face oval to remove background artifacts
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
            face_mask = cv.morphologyEx(face_mask, cv.MORPH_ERODE, kernel)

            # Masking out eyes and lips
            face_mask[le_mask] = 0
            face_mask[re_mask] = 0
            face_mask[lip_mask] = 0
                        
            # Cleaning up mask with morphological operations
            face_mask = cv.morphologyEx(face_mask, cv.MORPH_OPEN, kernel)
            face_mask = cv.morphologyEx(face_mask, cv.MORPH_CLOSE, kernel)

            # Shaving any remaining background artifacts with thresholds
            f = np.zeros_like(frame, dtype=np.uint8)
            f[face_mask == 255] = frame[face_mask == 255]
            white_mask = cv.inRange(cv.cvtColor(f, cv.COLOR_BGR2GRAY), 220, 255)
            face_mask[white_mask == 255] = 0

            dt = capture.get(cv.CAP_PROP_POS_MSEC)/1000
            
            if sat_only:
                if dt < onset_t:
                    result.write(frame)
                elif dt < offset_t:
                    weight = timing_func(dt)
                    frame_coloured = shift_color_temp(img=frame, img_mask=face_mask, shift_weight=weight, max_sat_shift=max_sat_shift, sat_only=True)
                    result.write(frame_coloured)
                else:
                    dt = cap_duration - dt
                    weight = timing_func(dt)
                    frame_coloured = shift_color_temp(img=frame, img_mask=face_mask, shift_weight=weight, max_sat_shift=max_sat_shift, sat_only=True) 
                    result.write(frame_coloured)
            else:
                if dt < onset_t:
                    result.write(frame)
                elif dt < offset_t:
                    weight = timing_func(dt)
                    frame_coloured = shift_color_temp(img=frame, img_mask=face_mask, shift_weight=weight, shift_color=shift_color, max_color_shift=max_color_shift, max_sat_shift=max_sat_shift)
                    result.write(frame_coloured)
                else:
                    dt = cap_duration - dt
                    weight = timing_func(dt)
                    frame_coloured = shift_color_temp(img=frame, img_mask=face_mask, shift_weight=weight, shift_color=shift_color, max_color_shift=max_color_shift, max_sat_shift=max_sat_shift)
                    result.write(frame_coloured)

        capture.release()
        result.release()