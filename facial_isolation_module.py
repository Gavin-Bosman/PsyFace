import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Defining pertinent facemesh landmark sets
LEFT_EYE_IDX = [301, 334, 296, 336, 285, 413, 464, 453, 452, 451, 450, 449, 448, 261, 265, 383, 301]
RIGHT_EYE_IDX = [71, 105, 66, 107, 55, 189, 244, 233, 232, 231, 230, 229, 228, 31, 35, 156, 71]
LIPS_IDX = [164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57, 186, 92, 165, 167, 164]
FACE_OVAL_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 345, 352, 376, 433, 397, 365, 379, 378, 400, 377, 
            152, 148, 176, 149, 150, 136, 172, 213, 147, 123, 116, 127, 162, 21, 54, 103, 67, 109, 10]

def create_path(landmark_set):
    """Creates a list of interconnected points, given a set of facial landmark indicies.
    
    Args: 
        landmark_set: a python list containing facial landmark indicies.
    
    Returns:
        routes: a list of tuples containing overlapping points, forming a path."""
    
    # Connvert the input list to a two-column dataframe
    landmark_dataframe = pd.DataFrame([(landmark_set[i], landmark_set[i+1]) for i in range(len(landmark_set) - 1)], columns=['p1', 'p2'])
    routes = []

    # Initialise the first two points
    p1 = landmark_dataframe.iloc[0]['p1']
    p2 = landmark_dataframe.iloc[0]['p2']

    for i in range(0, landmark_dataframe.shape[0]):
        obj = landmark_dataframe[landmark_dataframe['p1'] == p2]
        p1 = obj['p1'].values[0]
        p2 = obj['p2'].values[0]

        current_route = [p1, p2]
        routes.append(current_route)
    
    return routes

LEFT_EYE_PATH = create_path(LEFT_EYE_IDX)
RIGHT_EYE_PATH = create_path(RIGHT_EYE_IDX)
LIPS_PATH = create_path(LIPS_IDX)
FACE_OVAL_PATH = create_path(FACE_OVAL_IDX)

FACE_OVAL = 1
FACE_SKIN_ISOLATION = 2 
MASK_OPTIONS = [FACE_OVAL, FACE_SKIN_ISOLATION]

COLOR_SPACE_RGB = cv.COLOR_BGR2RGB
COLOR_SPACE_HSV = cv.COLOR_BGR2HSV
COLOR_SPACE_GRAYSCALE = cv.COLOR_BGR2GRAY
COLOR_SPACES = [COLOR_SPACE_RGB, COLOR_SPACE_HSV, COLOR_SPACE_GRAYSCALE]

def mask_face_region(inputDirectory, outputDirectory, maskType = FACE_SKIN_ISOLATION, withSubDirectories = False,
                     extractColorInfo = False, colorSpace = COLOR_SPACE_RGB):
    """Processes video files contained within inputDirectory with selected mask of choice.

    Args:
        inputDirectory: String
            A path string of the directory containing videos to process.

        outputDirectory: String
            A path string of the directory where processed videos will be written to.

        maskType: Integer
            An integer indicating the type of mask to apply to the input videos. This can be one of two options:
            either 1 for FACE_OVAL, or 2 for FACE_SKIN_ISOLATION.

        withSubDirectories: Boolean, 
            Indicating if the input directory contains subfolders.

        extractColorInfo: Boolean 
            Indicating if mean pixel values should be recorded and output in csv format.

        colorSpace: Integer
            Indicating which color space to operate in.
    
    Raises:
        ValueError: given invalid pathstrings or an unknown mask type.
    
    """

    global MASK_OPTIONS
    global FACE_OVAL
    global FACE_SKIN_ISOLATION
    global LEFT_EYE_PATH
    global RIGHT_EYE_PATH
    global LIPS_PATH
    global FACE_OVAL_PATH
    global COLOR_SPACE_RGB
    global COLOR_SPACE_HSV
    global COLOR_SPACE_GRAYSCALE
    global COLOR_SPACES
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                                                min_detection_confidence = 0.25, min_tracking_confidence = 0.75)

    # Type and value checks for function parameters
    if not isinstance(inputDirectory, str):
        raise TypeError("mask_face_region: invalid type for parameter inputDirectory.")
    elif not os.path.exists(inputDirectory):
        raise ValueError("mask_face_region: input directory path is not a valid path, or the directory does not exist.")
    
    if not isinstance(outputDirectory, str):
        raise TypeError("mask_face_region: invalid type for parameter outputDirectory.")
    elif not os.path.exists(outputDirectory):
        raise ValueError("mask_face_region: output directory path is not a valid path, or the directory does not exist.")
    
    if maskType not in MASK_OPTIONS:
        raise ValueError("mask_face_region: maskType must be either 1: indicating FACE_OVAL, or 2: indicating FACE_SKIN_ISOLATION.")

    if colorSpace not in COLOR_SPACES:
        raise ValueError("mask_face_region: colorSpace must match one of COLOR_SPACE_RGB, COLOR_SPACE_HSV, COLOR_SPACE_GRAYSCALE.")
    
    if not isinstance(withSubDirectories, bool):
        raise TypeError("mask_face_region: invalid type for parameter withSubDirectories.")
    
    if not isinstance(extractColorInfo, bool):
        raise TypeError("mask_face_region: invalid type for parameter extractColorInfo.")

    # Creating a list of file names to iterate through when processing
    files_to_process = []
    if not withSubDirectories:
         files_to_process = os.listdir(inputDirectory)
    else:
        ### TODO remove if file[0:2] ... after processing
        files_to_process = [os.path.join(path, file) 
                            for path, dirs, files in os.walk(inputDirectory, topdown=True) 
                            for file in files if file[0:2] == "02"]
    
    # Creating named output directories for video and csv output
    if not os.path.isdir(outputDirectory + "\\Video_Output"):
        os.mkdir(outputDirectory + "\\Video_Output")
    if not os.path.isdir(outputDirectory + "\\CSV_Output"):
        os.mkdir(outputDirectory + "\\CSV_Output")

    if maskType == FACE_SKIN_ISOLATION:

        for file in files_to_process:

            # Initialize capture and writer objects
            filename, extension = os.path.splitext(os.path.basename(file))
            capture = cv.VideoCapture(file)
            size = (int(capture.get(3)), int(capture.get(4)))
            result = cv.VideoWriter(outputDirectory + "\\Video_Output\\" + filename + "_masked.mp4",
                                    cv.VideoWriter.fourcc(*'MP4V'), 30, size)
            csv = None
            
            if extractColorInfo == True:
                if colorSpace == COLOR_SPACE_RGB:
                    csv = open(outputDirectory + "\\CSV_Output\\" + filename + "_RGB.csv", "w")
                    csv.write("Timestamp,Red,Green,Blue\n")
                elif colorSpace == COLOR_SPACE_HSV:
                    csv = open(outputDirectory + "\\CSV_Output\\" + filename + "_HSV.csv", "w")
                    csv.write("Timestamp,Hue,Saturation,Value\n")
                elif colorSpace == COLOR_SPACE_GRAYSCALE:
                    csv = open(outputDirectory + "\\CSV_Output\\" + filename + "_GRAYSCALE.csv", "w")
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

                # Extracting color values and writing to csv
                if extractColorInfo == True:
                    bin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    bin_mask[oval_mask] = 255
                    bin_mask[le_mask] = 0
                    bin_mask[re_mask] = 0
                    bin_mask[lip_mask] = 0

                    if colorSpace == COLOR_SPACE_RGB:
                        blue, green, red, *_ = cv.mean(frame, bin_mask)
                        timestamp = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                        csv.write(f"{timestamp:.5f},{red:.5f},{green:.5f},{blue:.5f}\n")

                    elif colorSpace == COLOR_SPACE_HSV:
                        hue, sat, val, *_ = cv.mean(cv.cvtColor(frame, colorSpace), bin_mask)
                        timestamp = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                        csv.write(f"{timestamp:.5f},{hue:.5f},{sat:.5f},{val:.5f}\n")
                    
                    elif colorSpace == COLOR_SPACE_GRAYSCALE:
                        val, *_ = cv.mean(cv.cvtColor(frame, colorSpace), bin_mask)
                        timestamp = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                        csv.write(f"{timestamp:.5f},{val:.5f}\n")

                if cv.waitKey(20) == ord('x'):
                    cv.destroyAllWindows()
                    break
        
            capture.release()
            result.release()
            csv.close()
    
    elif maskType == FACE_OVAL:

        for file in files_to_process:

            # Initializing capture and writer objects
            filename, extension = os.path.splitext(file)
            capture = cv.VideoCapture(inputDirectory + "\\" + file)
            size = (int(capture.get(3)), int(capture.get(4)))
            result = cv.VideoWriter(outputDirectory + "\\" + filename + "_masked" + extension,
                                    cv.VideoWriter.fourcc(*'MP4V'), 30, size)
            csv = None

            if extractColorInfo == True:
                if colorSpace == COLOR_SPACE_RGB:
                    csv = open(outputDirectory + "\\" + filename + "_RGB.csv", "w")
                    csv.write("Timestamp,Red,Green,Blue\n")
                elif colorSpace == COLOR_SPACE_HSV:
                    csv = open(outputDirectory + "\\" + filename + "_HSV.csv", "w")
                    csv.write("Timestamp,Hue,Saturation,Value\n")
                elif colorSpace == COLOR_SPACE_GRAYSCALE:
                    csv = open(outputDirectory + "\\" + filename + "_GRAYSCALE.csv", "w")
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

                # Extracting color values and writing to csv
                if extractColorInfo == True:
                    bin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    bin_mask[oval_mask] = 255

                    if colorSpace == COLOR_SPACE_RGB:
                        blue, green, red, *_ = cv.mean(frame, bin_mask)
                        timestamp = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                        csv.write(f"{timestamp:.5f},{red:.5f},{green:.5f},{blue:.5f}\n")

                    elif colorSpace == COLOR_SPACE_HSV:
                        hue, sat, val, *_ = cv.mean(cv.cvtColor(frame, colorSpace), bin_mask)
                        timestamp = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                        csv.write(f"{timestamp:.5f},{hue:.5f},{sat:.5f},{val:.5f}\n")
                    
                    elif colorSpace == COLOR_SPACE_GRAYSCALE:
                        val, *_ = cv.mean(cv.cvtColor(frame, colorSpace), bin_mask)
                        timestamp = capture.get(cv.CAP_PROP_POS_MSEC)/1000
                        csv.write(f"{timestamp:.5f},{val:.5f}\n")

                if cv.waitKey(20) == ord('x'):
                    cv.destroyAllWindows()
                    break
        
            capture.release()
            result.release()
            csv.close()