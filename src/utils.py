import cv2 as cv
import pandas as pd
import numpy as np
import os

# Defining pertinent facemesh landmark sets
LEFT_EYE_BROW_IDX = [301, 334, 296, 336, 285, 413, 464, 453, 452, 451, 450, 449, 448, 261, 265, 383, 301]
LEFT_EYE_IDX = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263]
LEFT_CHEEK_IDX = [265, 261, 448, 449, 450, 451, 452, 350, 277, 371, 266, 425, 280, 346, 340, 265]
RIGHT_EYE_BROW_IDX = [71, 105, 66, 107, 55, 189, 244, 233, 232, 231, 230, 229, 228, 31, 35, 156, 71]
RIGHT_EYE_IDX = [33, 7, 163, 144, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_CHEEK_IDX = [35, 31, 228, 229, 230, 231, 232, 233, 128, 114, 126, 142, 36, 205, 50, 117, 111, 35]
LIPS_IDX = [164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57, 186, 92, 165, 167, 164]
LIPS_TIGHT_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 0, 37, 39, 40, 185, 61]
FACE_OVAL_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 
                 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
FACE_OVAL_TIGHT_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 345, 352, 376, 433, 397, 365, 379, 378, 400, 377, 
            152, 148, 176, 149, 150, 136, 172, 213, 147, 123, 116, 127, 162, 21, 54, 103, 67, 109, 10]

def create_path(landmark_set:list[int]) -> list[tuple]:
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

# Preconstructed face region paths for use with facial manipulation functions
LEFT_EYE_BROW_PATH = create_path(LEFT_EYE_BROW_IDX)
LEFT_EYE_PATH = create_path(LEFT_EYE_IDX)
LEFT_CHEEK_PATH = create_path(LEFT_CHEEK_IDX)
RIGHT_EYE_BROW_PATH = create_path(RIGHT_EYE_BROW_IDX)
RIGHT_EYE_PATH = create_path(RIGHT_EYE_IDX)
RIGHT_CHEEK_PATH = create_path(RIGHT_CHEEK_IDX)
LIPS_PATH = create_path(LIPS_IDX)
LIPS_TIGHT_PATH = create_path(LIPS_TIGHT_IDX)
FACE_OVAL_PATH = create_path(FACE_OVAL_IDX)
FACE_OVAL_TIGHT_PATH = create_path(FACE_OVAL_TIGHT_IDX)

# Masking options for mask_face_region
FACE_OVAL = 1
FACE_OVAL_TIGHT = 2
FACE_SKIN_ISOLATION = 3
MASK_OPTIONS = [FACE_OVAL, FACE_OVAL_TIGHT, FACE_SKIN_ISOLATION]

# Compatible color spaces for extract_color_channel_means and face_color_shift
COLOR_SPACE_RGB = cv.COLOR_BGR2RGB
COLOR_SPACE_HSV = cv.COLOR_BGR2HSV
COLOR_SPACE_GRAYSCALE = cv.COLOR_BGR2GRAY
COLOR_SPACES = [COLOR_SPACE_RGB, COLOR_SPACE_HSV, COLOR_SPACE_GRAYSCALE]

COLOR_RED = 4
COLOR_BLUE = 5
COLOR_GREEN = 6
COLOR_YELLOW = 7

def get_min_max_rgb(filePath:str, focusColor:int|str = COLOR_RED) -> tuple:
    """Given an input video file path, returns the minimum and maximum (B,G,R) colors, containing the minimum and maximum
        values of the focus color. 
    
        Args:
            filePath: String
                The path string of the location of the file to be processed.
            
            focusColor: int | String
                The RGB color channel to focus on. Either one of the predefined color constants, or a string literal of the colors name.
        
        Raises:
            TypeError: given invalid parameter types.
            ValueError: given a nonexisting file path, or a non RGB focus color.
        
        Returns:
            min_color: ndarray[int]
            max_color: ndarray[int]
    """

    global COLOR_RED
    global COLOR_BLUE
    global COLOR_GREEN

    # Type and value checking before computation
    if not isinstance(filePath, str):
        raise TypeError("get_min_max_rgb: invalid type for filePath.")
    elif not os.path.exists(filePath):
        raise ValueError("get_min_max_rgb: filePath not a valid path.")
    
    if isinstance(focusColor, str):
        if str.lower(focusColor) not in ["red", "green", "blue"]:
            raise ValueError("get_min_max_rgb: focusColor not a valid color option.")
    elif isinstance(focusColor, int):
        if focusColor not in [COLOR_RED, COLOR_BLUE, COLOR_GREEN]:
            raise ValueError("get_min_max_rgb: focusColor not a valid color option.")
    else:
        raise TypeError("get_min_max_rgb: invalid type for focusColor.")

    capture = cv.VideoCapture(filePath)
    if not capture.isOpened():
        print("get_min_max_rgb: Error opening videoCapture object.")
        return -1

    min_x, min_y, max_x, max_y, min_color, max_color = 0,0,0,0,None,None
    min_val, max_val = 255, 0

    while True:

        success, frame = capture.read()
        if not success:
            break

        blue, green, red = cv.split(frame)

        if focusColor == COLOR_RED or str.lower(focusColor) == "red":
            max_y = np.where(red == red.max())[0][0]
            max_x = np.where(red == red.max())[1][0]
            cur_max_val = red[max_y, max_x]

            min_y = np.where(red == red.min())[0][0]
            min_x = np.where(red == red.min())[1][0]
            cur_min_val = red[min_y, min_x]

            if cur_max_val > max_val:
                max_val = cur_max_val
                max_color = frame[max_y, max_x]

            if cur_min_val < min_val:
                min_val = cur_min_val
                min_color = frame[min_y, min_x]

        elif focusColor == COLOR_BLUE or str.lower(focusColor) == "blue":
            max_y = np.where(blue == blue.max())[0][0]
            max_x = np.where(blue == blue.max())[1][0]
            cur_max_val = blue[max_y, max_x]

            min_y = np.where(blue == blue.min())[0][0]
            min_x = np.where(blue == blue.min())[1][0]
            cur_min_val = blue[min_y, min_x]

            if cur_max_val > max_val:
                max_val = cur_max_val
                max_color = frame[max_y, max_x]

            if cur_min_val < min_val:
                min_val = cur_min_val
                min_color = frame[min_y, min_x]
        
        else:
            max_y = np.where(green == green.max())[0][0]
            max_x = np.where(green == green.max())[1][0]
            cur_max_val = green[max_y, max_x]

            min_y = np.where(green == green.min())[0][0]
            min_x = np.where(green == green.min())[1][0]
            cur_min_val = green[min_y, min_x]

            if cur_max_val > max_val:
                max_val = cur_max_val
                max_color = frame[max_y, max_x]

            if cur_min_val < min_val:
                min_val = cur_min_val
                min_color = frame[min_y, min_x]
    
    return (min_color, max_color)

# Defining useful timing functions
def sigmoid(x:float, k:float = 1.0) -> float:
    return 1/(1 + np.exp(-k * x))