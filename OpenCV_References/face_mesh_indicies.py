import cv2
import mediapipe as mp
import numpy as np

# Variables to keep track of zoom level and coordinates
zoom_factor = 1.0
zoom_increment = 0.1
x_offset, y_offset = 0, 0
drag_start_x, drag_start_y = None, None
is_dragging = False

# Initialize MediaPipe Face Mesh Task
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# read in the image, feel free to replace with any portrait image
image = cv2.imread('OpenCV_References/Photos/portrait_2.jpg')
height, width, *_ = image.shape
dims = (int(width*2.5), int(height*2.5))
image = cv2.resize(image, dims, interpolation=cv2.INTER_LINEAR)
original_image = image.copy()

# Perform face landmark detection
results = face_mesh.process(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

# Draw landmarks if detected
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Extract coordinates of facial landmarks
        h, w, _ = original_image.shape
        landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

        # Draw landmarks
        for idx, (x, y) in enumerate(landmarks):
            cv2.circle(original_image, (x, y), 1, (0, 255, 0), -1)
            cv2.putText(original_image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1, cv2.LINE_AA)

def update_zoomed_image():
    global image, zoom_factor, x_offset, y_offset
    height, width = original_image.shape[:2]

    # Calculate the size of the zoomed window
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)

    # Ensure the offsets do not go out of the image bounds
    x_offset = min(max(0, x_offset), width - new_width)
    y_offset = min(max(0, y_offset), height - new_height)

    # Extract the zoomed region
    zoomed_image = original_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width]

    # Resize the zoomed image to the original size for display
    image = cv2.resize(zoomed_image, (width, height))

def mouse_callback(event, x, y, flags, param):
    global zoom_factor, x_offset, y_offset, drag_start_x, drag_start_y, is_dragging

    if event == cv2.EVENT_MOUSEWHEEL:
        # Zoom in or out with the mouse wheel
        if flags > 0:
            zoom_factor = min(zoom_factor + zoom_increment, 5.0)
        else:
            zoom_factor = max(zoom_factor - zoom_increment, 1.0)
        update_zoomed_image()

    elif event == cv2.EVENT_LBUTTONDOWN:
        # Start dragging
        is_dragging = True
        drag_start_x, drag_start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        # Update the offsets while dragging
        dx = int((x - drag_start_x) / zoom_factor)
        dy = int((y - drag_start_y) / zoom_factor)
        x_offset = max(0, min(x_offset - dx, original_image.shape[1] - int(original_image.shape[1] / zoom_factor)))
        y_offset = max(0, min(y_offset - dy, original_image.shape[0] - int(original_image.shape[0] / zoom_factor)))
        drag_start_x, drag_start_y = x, y
        update_zoomed_image()

    elif event == cv2.EVENT_LBUTTONUP:
        # End dragging
        is_dragging = False

# Create a named window and set the mouse callback
cv2.namedWindow('Zoomable Image')
cv2.setMouseCallback('Zoomable Image', mouse_callback)

update_zoomed_image()

while True:
    cv2.imshow('Zoomable Image', image)

    if cv2.waitKey(5) == ord('x'):  # 'x' key to exit
        break

cv2.destroyAllWindows()
