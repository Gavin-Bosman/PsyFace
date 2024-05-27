import cv2 as cv
import mediapipe as mp
import pandas as pd
import numpy as np

# initialise videocapture object
capture = cv.VideoCapture('Video_Song_Actor_01/Actor_01/02-02-01-01-01-02-01.mp4')

# initialise videowriter object
size = (int(capture.get(3)), int(capture.get(4)))
result = cv.VideoWriter('Video_Song_Actor_01_Masked/02-02-01-01-01-02-01-masked.mp4', cv.VideoWriter.fourcc(*'MP4V'), 20.0, size)

# Initialising the mediapipe task
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
connections = mp.solutions.face_mesh_connections

# landmark indicies of the left eye region
le_idx = [301, 334, 296, 336, 285, 413, 464, 453, 452, 451, 450, 449, 448, 261, 265, 383, 301]
# landmark indicies of the right eye region
re_idx = [71, 105, 66, 107, 55, 189, 244, 233, 232, 231, 230, 229, 228, 31, 35, 156, 71]
# landmark indicies of the lips
lips_idx = [164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57, 186, 92, 165, 167, 164]
# landmark indicies of the face oval
face_idx = [10, 338, 297, 332, 284, 251, 389, 356, 345, 352, 376, 433, 397, 365, 379, 378, 400, 377, 
            152, 148, 176, 149, 150, 136, 172, 213, 147, 123, 116, 127, 162, 21, 54, 103, 67, 109, 10]

# convert to two col dataframe
left_eye = pd.DataFrame([(le_idx[i], le_idx[i+1]) for i in range(len(le_idx) - 1)], columns=['p1', 'p2'])
right_eye = pd.DataFrame([(re_idx[i], re_idx[i+1]) for i in range(len(re_idx) - 1)], columns=['p1', 'p2'])
lips = pd.DataFrame([(lips_idx[i], lips_idx[i+1]) for i in range(len(lips_idx)-1)], columns=['p1', 'p2'])
face_oval = pd.DataFrame([(face_idx[i], face_idx[i+1]) for i in range(len(face_idx)-1)], columns=['p1', 'p2'])

def create_path(landmark_dataframe):
    routes = []

    # initialise the first points
    p1 = landmark_dataframe.iloc[0]['p1']
    p2 = landmark_dataframe.iloc[0]['p2']

    for i in range(0, landmark_dataframe.shape[0]):
        obj = landmark_dataframe[landmark_dataframe['p1'] == p2]
        p1 = obj['p1'].values[0]
        p2 = obj['p2'].values[0]

        current_route = [p1, p2]
        routes.append(current_route)
    
    return routes

# preprocessing to not bog up between frame computation
left_eye = create_path(left_eye)
right_eye = create_path(right_eye)
lips = create_path(lips)
face_oval = create_path(face_oval)

while True:
    success, frame = capture.read()

    # Mediapipe makes use of RGB while cv2 uses BGR
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    landmark_coords = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # display the actual face mesh with line of code below:
            # mpDraw.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=mpFaceMesh.FACEMESH_TESSELATION)

            # The multi_face_landmarks are provided in normalised [0,1] coordinates, convert to cartesian
            for id,lm in enumerate(face_landmarks.landmark):
                ih, iw, ic = frame.shape
                x,y = int(lm.x * iw), int(lm.y * ih)
                landmark_coords.append({'id':id, 'x':x, 'y':y})
        
    # create facial mask
    le_screen_coords = []
    re_screen_coords = []
    lips_screen_coords = []
    face_outline_coords = []
    
    # left eye screen coordinates
    for cur_source, cur_target in left_eye:
        source = landmark_coords[cur_source]
        target = landmark_coords[cur_target]
        le_screen_coords.append((source.get('x'),source.get('y')))
        le_screen_coords.append((target.get('x'),target.get('y')))
    
    # right eye screen coordinates
    for cur_source, cur_target in right_eye:
        source = landmark_coords[cur_source]
        target = landmark_coords[cur_target]
        re_screen_coords.append((source.get('x'),source.get('y')))
        re_screen_coords.append((target.get('x'),target.get('y')))

    # lips screen coordinates
    for cur_source, cur_target in lips:
        source = landmark_coords[cur_source]
        target = landmark_coords[cur_target]
        lips_screen_coords.append((source.get('x'),source.get('y')))
        lips_screen_coords.append((target.get('x'),target.get('y')))
    
    # face oval screen coordinates
    for cur_source, cur_target in face_oval:
        source = landmark_coords[cur_source]
        target = landmark_coords[cur_target]
        face_outline_coords.append((source.get('x'),source.get('y')))
        face_outline_coords.append((target.get('x'),target.get('y')))

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

    # removing eyes, brows, and lips
    masked_frame = frame
    masked_frame[le_mask] = 0
    masked_frame[re_mask] = 0
    masked_frame[lip_mask] = 0

    # last step, masking out the bounding face shape
    face_skin = np.zeros_like(masked_frame)
    face_skin[oval_mask] = masked_frame[oval_mask] 

    bin_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    bin_mask[oval_mask] = 255
    bin_mask[le_mask] = 0
    bin_mask[re_mask] = 0
    bin_mask[lip_mask] = 0

    # later will store these BGR values in some tabular format for statistical analysis
    print(cv.mean(frame, bin_mask))

    # result.write(face_skin)
    cv.imshow('Video', face_skin)

    # The cv.waitkey() parameter changes the resulting frame rate, increase or decrease as needed
    if cv.waitKey(30) == ord('x'):
        # close the window on 'x' keypress
        cv.destroyAllWindows()
        break

capture.release()
# result.release()