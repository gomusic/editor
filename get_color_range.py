import cv2
import numpy as np

# Initialize the global variable to store the color selected by mouse click
clicked_color_hsv = None
current_frame = None

# Variables for panning (dragging) the view
pan_x, pan_y = 0, 0
is_dragging = False
start_x, start_y = 0, 0  # Initial position when starting to drag
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the BGR and HSV pixel values at the clicked point
        pixel_bgr = current_frame[y, x, :]
        pixel_hsv = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_BGR2HSV)
        pixel_hsv = pixel_hsv[0][0]

        # Print the BGR and HSV values
        print("Clicked pixel BGR:", pixel_bgr)
        print("Clicked pixel HSV:", pixel_hsv)

def mouse_drag(event, x, y, flags, param):
    global pan_x, pan_y, is_dragging, start_x, start_y
    if event == cv2.EVENT_MBUTTONDOWN:  # Middle mouse button pressed
        is_dragging = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:  # Mouse moving while dragging
        # Calculate the shift in mouse position
        dx = x - start_x
        dy = y - start_y
        start_x, start_y = x, y  # Update start positions for smooth drag

        # Update pan_x and pan_y, ensuring we stay within the image bounds
        pan_x = min(max(0, pan_x - dx), current_frame.shape[1] - window_width)
        pan_y = min(max(0, pan_y - dy), current_frame.shape[0] - window_height)

    elif event == cv2.EVENT_MBUTTONUP:  # Middle mouse button released
        is_dragging = False


def mouse_event_handler(event, x, y, flags, param):
    global clicked_color_hsv, pan_x, pan_y, is_dragging, start_x, start_y

    # Left-click to select color
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the BGR and HSV pixel values at the clicked point, considering panning
        pixel_bgr = current_frame[y + pan_y, x + pan_x, :]
        pixel_hsv = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_BGR2HSV)
        clicked_color_hsv = pixel_hsv[0][0]

        # Print the BGR and HSV values
        print("Clicked pixel BGR:", pixel_bgr)
        print("Clicked pixel HSV:", clicked_color_hsv)

    # Middle-click to start dragging
    elif event == cv2.EVENT_MBUTTONDOWN:
        is_dragging = True
        start_x, start_y = x, y

    # Mouse move while dragging
    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        dx = x - start_x
        dy = y - start_y
        start_x, start_y = x, y

        # Update pan_x and pan_y, ensuring we stay within the image bounds
        pan_x = min(max(0, pan_x - dx), current_frame.shape[1] - window_width)
        pan_y = min(max(0, pan_y - dy), current_frame.shape[0] - window_height)

    # Middle button released, stop dragging
    elif event == cv2.EVENT_MBUTTONUP:
        is_dragging = False


def main():
    global current_frame
    cap = cv2.VideoCapture('com.mov')

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', mouse_click)

    while True:
        ret, frame = cap.read()
        current_frame = frame
        if not ret:
            break



        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_color(frame):
    global current_frame, pan_x, pan_y, window_width, window_height

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', mouse_event_handler)

    # Screen dimensions for displaying the portion of the frame
    screen_height, screen_width = 720, 1280  # Set desired visible window size
    window_height, window_width = screen_height, screen_width

    while True:
        current_frame = frame

        # Calculate the visible frame section based on panning offsets
        visible_frame = frame[pan_y:pan_y + window_height, pan_x:pan_x + window_width]

        # Display the visible section of the frame
        cv2.imshow('frame', visible_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()
