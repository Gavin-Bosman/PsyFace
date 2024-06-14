### TODO add functionality for folders with subdirectories
### TODO add face oval masking to mask_face_region
### TODO add function for writing colour codes in both rgb and hsv

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


def mask_face_region(inputDirectory, outputDirectory, maskType = FACE_SKIN_ISOLATION, withSubDirectories = False):
    """Processes video files contained within inputDirectory with selected mask of choice.

    Args:
        inputDirectory: a path string of the directory containing videos to process.
        outputDirectory: a path string of the directory where processed videos will be written to.
        maskType: an integer indicating the type of mask to apply to the input videos. This can be one of two options:
            either 1 for FACE_OVAL, or 2 for FACE_SKIN_ISOLATION.
        withSubDirectories: a boolean, indicating if the input directory contains subfolders.
    
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
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces = 1, static_image_mode = False, 
                                                min_detection_confidence = 0.25, min_tracking_confidence = 0.75)

    if maskType not in MASK_OPTIONS:
        raise ValueError("mask_face_region: maskType must be either 1: indicating FACE_OVAL, or 2: indicating FACE_SKIN_ISOLATION.")
    
    if not os.path.exists(inputDirectory):
        raise ValueError("mask_face_region: input directory path is not a valid path, or the directory does not exist.")
    
    if not os.path.exists(outputDirectory):
        raise ValueError("mask_face_region: output directory path is not a valid path, or the directory does not exist.")
    
    if not withSubDirectories:
        if maskType == FACE_SKIN_ISOLATION:
            files_to_process = os.listdir(inputDirectory)

            for file in files_to_process:

                filename, extension = os.path.splitext(file)
                capture = cv.VideoCapture(inputDirectory + "\\" + file)
                size = (int(capture.get(3)), int(capture.get(4)))
                result = cv.VideoWriter(outputDirectory + "\\" + filename + "_masked" + extension,
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

                    if cv.waitKey(20) == ord('x'):
                        cv.destroyAllWindows()
                        break
            
                capture.release()
                result.release()
