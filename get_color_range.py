import cv2
import numpy as np

# Initialize the global variable to store the color selected by mouse click
clicked_color_hsv = None
current_frame = None
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the BGR and HSV pixel values at the clicked point
        pixel_bgr = current_frame[y, x, :]
        pixel_hsv = cv2.cvtColor(np.uint8([[pixel_bgr]]), cv2.COLOR_BGR2HSV)
        pixel_hsv = pixel_hsv[0][0]

        # Print the BGR and HSV values
        print("Clicked pixel BGR:", pixel_bgr)
        print("Clicked pixel HSV:", pixel_hsv)

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

if __name__ == '__main__':
    main()
