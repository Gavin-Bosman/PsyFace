import cv2 as cv
import cv2.typing
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from typing import Callable
from utils import *

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
        ValueError: given invalid pathstrings or an unknown mask type.
        TypeError: given invalid parameter types.
    
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
        raise TypeError("mask_face_region: invalid type for parameter inputDirectory.")
    elif not os.path.exists(input_dir):
        raise ValueError("mask_face_region: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(input_dir):
        singleFile = True
    
    if not isinstance(output_dir, str):
        raise TypeError("mask_face_region: invalid type for parameter outputDirectory.")
    elif not os.path.exists(output_dir):
        raise ValueError("mask_face_region: output directory path is not a valid path, or the directory does not exist.")
    
    if mask_type not in MASK_OPTIONS:
        raise ValueError("mask_face_region: maskType must be either 1: indicating FACE_OVAL, or 2: indicating FACE_SKIN_ISOLATION.")
    
    if not isinstance(with_sub_dirs, bool):
        raise TypeError("mask_face_region: invalid type for parameter withSubDirectories.")

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
    
    # Creating named output directories for video and csv output
    if not os.path.isdir(output_dir + "\\Video_Output"):
        os.mkdir(output_dir + "\\Video_Output")

    if mask_type == FACE_SKIN_ISOLATION:

        for file in files_to_process:

            # Initialize capture and writer objects
            filename, extension = os.path.splitext(os.path.basename(file))
            capture = cv.VideoCapture(file)
            if not capture.isOpened():
                print("mask_face_region: Error opening videoCapture object.")
                return -1

            size = (int(capture.get(3)), int(capture.get(4)))
            result = cv.VideoWriter(output_dir + "\\Video_Output\\" + filename + "_masked.mp4",
                                    cv.VideoWriter.fourcc(*'MP4V'), 30, size)
            
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
                face_skin = np.zeros_like(masked_frame)
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
                print("mask_face_region: Error opening videoCapture object.")
                return -1

            size = (int(capture.get(3)), int(capture.get(4)))
            result = cv.VideoWriter(output_dir + "\\Video_Output\\" + filename + "_masked" + extension,
                                    cv.VideoWriter.fourcc(*'MP4V'), 30, size)
            
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
                print("mask_face_region: Error opening videoCapture object.")
                return -1

            size = (int(capture.get(3)), int(capture.get(4)))
            result = cv.VideoWriter(output_dir + "\\Video_Output\\" + filename + "_masked" + extension,
                                    cv.VideoWriter.fourcc(*'MP4V'), 30, size)
            
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
                                with_sub_dirs:bool = False, mask_face:bool = True):
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
        raise TypeError("extract_color_channel_means: input_dir must be a path string.")
    elif not os.path.exists(input_dir):
        raise OSError("extract_color_channel_means: input_dir is not a valid path.")
    elif os.path.isfile(input_dir):
        singleFile = True
    
    if not isinstance(output_dir, str):
        raise TypeError("extract_color_channel_means: output_dir must be a path string.")
    elif not os.path.exists(output_dir):
        raise OSError("extract_color_channel_means: output_dir is not a valid path.")
    elif not os.path.isdir(output_dir):
        raise OSError("extract_color_channel_means: output_dir must be a path string to a directory.")
    
    if not isinstance(color_space, int):
        if not isinstance(color_space, str):
            raise TypeError("extract_color_channel_means: color_space must be an int or str.")
    if isinstance(color_space, str):
        if str.lower(color_space) not in ["rgb", "hsv", "grayscale"]:
            raise ValueError("extract_color_channel_means: unspecified color space.")
        else:
            if str.lower(color_space) == "rgb":
                color_space = COLOR_SPACE_RGB
            if str.lower(color_space) == "hsv":
                color_space = COLOR_SPACE_HSV
            if str.lower(color_space) == "grayscale":
                color_space = COLOR_SPACE_GRAYSCALE

    if isinstance(color_space, int):
        if color_space not in [COLOR_SPACE_RGB, COLOR_SPACE_HSV, COLOR_SPACE_GRAYSCALE]:
            raise ValueError("extract_color_channel_means: unspecified color space.")
    
    if not isinstance(with_sub_dirs, bool):
        raise TypeError("extract_color_channel_means: with_sub_dirs must be a boolean.")

    if singleFile:
        files_to_process.append(input_dir)
    elif not with_sub_dirs:
         files_to_process = os.listdir(input_dir)
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(input_dir, topdown=True) 
                            for file in files]
    
    # create an output directory for the csv files
    if not os.path.isdir(output_dir + "\\CSV_Output"):
        os.mkdir(output_dir + "\\CSV_Output")
        output_dir = os.path.join(output_dir, "\\CSV_Output")
    
    for file in files_to_process:

        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        capture = cv.VideoCapture(file)
        if not capture.isOpened():
            print("extract_color_channel_means: Error opening videoCapture object.")
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

def color_shift(img: cv2.typing.MatLike, mask: cv2.typing.MatLike, weight: float, sat_delta: float = 0.0, 
                sat_only: bool = False, color: str|int = COLOR_RED, maxShift: float = 8.0) -> cv2.typing.MatLike:
    """Takes in an image and a mask of the same shape, and shifts the specified color temperature by weight units in the masked
        region of the image. This function makes use of the CIE Lab perceptually uniform color space to perform natural looking
        color shifts on the face.

        Args:
            img: Matlike
                An input still image or video frame.

            mask: Matlike
                A binary image with the same shape as img.

            weight: float
                The current shifting weight; a float in the range [0,1] returned from a timing function. 
            
            sat_delta: float
                The units to shift the images saturation.
            
            sat_only: bool
                A specifier that indicates if only the saturation is being modified.

            color: str | int
                An integer or string literal specifying which color will be applied to the input image.

            maxShift: float
                The maximum units to shift the colour temperature by, during peak onset.
            
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
        raise TypeError("color_shift: parameter img must be a Matlike.")
    if not isinstance(mask, cv2.typing.MatLike):
        raise TypeError("color_shift: parameter mask must be a Matlike.")
    if img.shape[:2] != mask.shape[:2]:
        raise ValueError("color_shift: image and mask have different shapes.")
    
    if not isinstance(weight, float):
        raise TypeError("color_shift: parameter weight must be of type float.")
    if not isinstance(sat_delta, float):
        raise TypeError("color_shift: parameter sat_delta must be of type float.")
    
    if isinstance(color, str):
        if str.lower(color) not in ["red", "green", "blue", "yellow"]:
            raise ValueError("color_shift: color must be one of: red, green, blue, yellow.")
    elif isinstance(color, int):
        if color not in [COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW]:
            raise ValueError("color_shift: color must be one of: red, green, blue, yellow.")
    else:
        raise TypeError("color_shift: parameter color must be of type str or int.")
    
    if not isinstance(maxShift, float):
        raise TypeError("color_shift: parameter maxShift must be of type float.")
    
    result = None

    if not sat_only:

        img_LAB = None

        if sat_delta != 0.0:
            img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)
            h,s,v = cv.split(img_hsv)
            s = np.where(mask == 255, s + (weight * sat_delta), s)
            np.clip(s,0,255)
            img_hsv = cv.merge([h,s,v])

            img_LAB = cv.cvtColor(img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)

        # Convert input image to CIE La*b* color space (perceptually uniform space)
        img_LAB = cv.cvtColor(img, cv.COLOR_BGR2LAB).astype(np.float32)
        l,a,b = cv.split(img_LAB)

        if color == COLOR_RED or str.lower(color) == "red":
            a = np.where(mask==255, a + (weight * maxShift), a)
            np.clip(a, -128, 127)
        if color == COLOR_BLUE or str.lower(color) == "blue":
            b = np.where(mask==255, b - (weight * maxShift), b)
            np.clip(a, -128, 127)
        if color == COLOR_GREEN or str.lower(color) == "green":
            a = np.where(mask==255, a - (weight * maxShift), a)
            np.clip(a, -128, 127)
        if color == COLOR_YELLOW or str.lower(color) == "yellow":
            b = np.where(mask==255, b + (weight * maxShift), b)
            np.clip(a, -128, 127)
        
        img_LAB = cv.merge([l,a,b])
        
        # Convert CIE La*b* back to BGR
        result = cv.cvtColor(img_LAB.astype(np.uint8), cv.COLOR_LAB2BGR)
        
    elif sat_delta != 0.0:
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)
        h,s,v = cv.split(img_hsv)
        s = np.where(mask == 255, s + (weight * sat_delta), s)
        np.clip(s,0,255)
        img_hsv = cv.merge([h,s,v])

        result = cv.cvtColor(img_hsv.astype(np.uint8), cv.COLOR_HSV2BGR)

    return result

def face_color_shift(inputDirectory:str, outputDirectory:str, onset:float, offset:float, maxShift: float = 8.0, sat_delta: float = 0.0,
                     timingFunc:Callable[...,float] = sigmoid, filterColor:str|int = COLOR_RED, withSubDirectories:bool = False) -> None: 
    """Takes in one or more videos contaned in inputDirectory, and applies a specified colour filter to the facial skin
        region. The resulting videos are output to outputDirectory.

        Args:
            inputDirectory: String
                A path string to the directory containing input video files.

            outputDirectory: String
                A path string to the directory where outputted video files will be saved.
            
            onset: float
                The onset time of the colour filtering.
            
            offset: float
                The offset time of the colour filtering.
            
            maxShift: float
                The maximum units to shift the colour temperature by, during peak onset.
            
            sat_delta: float
                The units to shift the images saturation.
            
            timingFunc: Function() -> float
                Any function that takes at least one input float (time), and returns a float.

            filterColor: String | int
                Either a string literal specifying the color of choice, or a predefined integer constant.
            
            withSubDirectories: bool
                A boolean value indicating whether the input directory contains nested directories.
        
        Throws:
            TypeError: given invalid parameter types.
            ValueError: given invalid directory paths, or alpha value outside the range [0,1].
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
    if not isinstance(inputDirectory, str):
        raise TypeError("face_color_shift: invalid type for parameter inputDirectory.")
    elif not os.path.exists(inputDirectory):
        raise ValueError("face_color_shift: input directory path is not a valid path, or the directory does not exist.")
    elif os.path.isfile(inputDirectory):
        singleFile = True
    
    if not isinstance(outputDirectory, str):
        raise TypeError("face_color_shift: invalid type for parameter outputDirectory.")
    elif not os.path.exists(outputDirectory):
        raise ValueError("face_color_shift: output directory path is not a valid path, or the directory does not exist.")
    elif not os.path.isdir(outputDirectory):
        raise ValueError("Face_color_shift: outputDirectory must be a valid path to a directory.")
    
    if not isinstance(onset, float):
        raise TypeError("face_color_shift: parameter onset must be a float.")
    if not isinstance(offset, float):
        raise TypeError("face_color_shift: parameter offset must be a float.")
    if not isinstance(maxShift, float):
        raise TypeError("face_color_shift: parameter maxShift must be a float.")
    if not isinstance(sat_delta, float):
        raise TypeError("face_color_shift: parameter sat_delta must be a float.")
    
    if isinstance(timingFunc, Callable):
        if not isinstance(timingFunc(1.0), float):
            raise ValueError("face_color_shift: timingFunc must return a float value.")
    # add check for return value in range [0,1]

    if isinstance(filterColor, str):
        if str.lower(filterColor) not in ["red", "green", "blue", "yellow"]:
            raise ValueError("Face_color_shift: color must be one of: red, green, blue, yellow.")
    elif isinstance(filterColor, int):
        if filterColor not in [COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_YELLOW]:
            raise ValueError("Face_color_shift: color must be one of: red, green, blue, yellow.")
    else:
        raise TypeError("Face_color_shift: color must be of type str or int.")

    if not isinstance(withSubDirectories, bool):
        raise TypeError("Face_color_shift: invalid type for parameter withSubDirectories.")

    # Creating a list of file names to iterate through when processing
    files_to_process = []

    if singleFile:
        files_to_process.append(inputDirectory)
    elif not withSubDirectories:
         files_to_process = os.listdir(inputDirectory)
    else:
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(inputDirectory, topdown=True) 
                            for file in files]
    
    for file in files_to_process:
            
        # Initialize capture and writer objects
        filename, extension = os.path.splitext(os.path.basename(file))
        capture = cv.VideoCapture(file)
        if not capture.isOpened():
            print("face_color_shift: Error opening video file.")
            return -1

        size = (int(capture.get(3)), int(capture.get(4)))
        result = cv.VideoWriter(outputDirectory + "\\" + filename + "_color_filter.mp4",
                                cv.VideoWriter.fourcc(*'MP4V'), 30, size)
        frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv.CAP_PROP_FPS)
        cap_duration = float(frame_count)/float(fps)
        print("Duration: ", cap_duration)
            
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
            face_mask = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            face_mask[oval_mask] = 255
            face_mask[le_mask] = 0
            face_mask[re_mask] = 0
            face_mask[lip_mask] = 0

            # Cleaning up mask with morphological operations
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
            face_mask = cv.morphologyEx(face_mask, cv.MORPH_OPEN, kernel)
            face_mask = cv.morphologyEx(face_mask, cv.MORPH_CLOSE, kernel)

            dt = capture.get(cv.CAP_PROP_POS_MSEC)/1000
            
            if dt < onset:
                result.write(frame)
            elif dt < offset:
                weight = timingFunc(dt)
                frame_coloured = color_shift(img=frame, mask=face_mask, weight=weight, color=filterColor, maxShift=maxShift, sat_delta=sat_delta)
                result.write(frame_coloured)
            else:
                dt = cap_duration - dt
                weight = timingFunc(dt)
                frame_coloured = color_shift(img=frame, mask=face_mask, weight=weight, color=filterColor, maxShift=maxShift, sat_delta=sat_delta)
                result.write(frame_coloured)

        capture.release()
        result.release()