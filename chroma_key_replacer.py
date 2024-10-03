import cv2
import numpy as np
import base64
from cvzone.SelfiSegmentationModule import SelfiSegmentation

segmentor = SelfiSegmentation(model=0)

# Load the main video
video = cv2.VideoCapture('com.mov')
pha_video = cv2.VideoCapture('pha.mov')

# Load the background video that will replace the phone screen
background_video = cv2.VideoCapture('tiktok_vertical.mov')

# Get the properties of the main video
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output_detected_video.mp4', fourcc, fps, (frame_width, frame_height))

# Define the blue color range in HSV for detecting the phone screen
lower_blue = np.array([80, 50, 80])  # adjust based on your blue color range
upper_blue = np.array([130, 255, 255])  # adjust based on your blue color range

lower_green = np.array([120, 255, 150])  # adjust based on your blue color range
upper_green = np.array([120, 255, 160])  # adjust based on your blue color range

# Initialize detection tracking
consecutive_frame_count = 0
detected_for_required_period = False
start_zooming = False
required_frames_for_one_second = fps  # Number of frames in 1 second

# Initialize zoom scale
zoom_scale = 1.0
zoom_increment = 0.2  # Adjust zoom speed

# Function to apply zoom towards the center of the background frame in the main frame
import cv2


def adjust_contrast(frame, alpha, beta=0):
    """
    Adjust the contrast and brightness of a frame.

    Parameters:
    - frame: Input image/frame (numpy array)
    - alpha: Contrast control (1.0 means no change, >1 increases contrast, <1 decreases contrast)
    - beta: Brightness control (default is 0 for no change)

    Returns:
    - Adjusted frame with new contrast and brightness.
    """
    # Apply the contrast and brightness adjustment
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    return adjusted_frame


def frame_to_base64(frame):
    # Encode the frame in memory as a JPEG
    _, buffer = cv2.imencode('.jpg', frame)

    # Convert the buffer to a base64 string
    base64_str = base64.b64encode(buffer).decode('utf-8')

    return base64_str


lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])


# Function for basic spill suppression (adjust colors near the edges)


def apply_zoom_to_center(main_frame, background_rect, background_frame):
    global zoom_scale

    h, w = main_frame.shape[:2]  # Main frame height and width
    rect_x, rect_y, rect_w, rect_h = background_rect  # Background rect position and dimensions

    # Ensure the zoom is centered on the background frame's center
    center_x = rect_x + rect_w // 2
    center_y = rect_y + rect_h // 2

    # Calculate the new size of the zoomed frame
    new_w = int(w / zoom_scale)
    new_h = int(h / zoom_scale)

    new_bg_w = int(rect_w * zoom_scale)
    new_bg_h = int(rect_h * zoom_scale)

    # Check if the zoomed background frame is large enough to cover the detected area
    if new_bg_w >= frame_width or new_bg_h >= frame_height:
        return background_frame

    # Calculate the coordinates for cropping the zoomed-in frame
    x1 = max(0, center_x - new_w // 2)
    y1 = max(0, center_y - new_h // 2)
    x2 = min(w, center_x + new_w // 2)
    y2 = min(h, center_y + new_h // 2)

    cropped_frame = main_frame[y1:y2, x1:x2]

    # Resize back to the original size (zoom effect)
    zoomed_frame = cv2.resize(cropped_frame, (w, h))

    return zoomed_frame


def apply_background(overlay_alpha, overlay_color, background_frame):
    main_background = cv2.resize(background_frame, (frame_width, frame_height))
    overlay_alpha = overlay_alpha / 255.0

    # Define where you want to overlay the image (top-left corner)
    x_offset, y_offset = 0, 0  # Change these values as needed

    # Get the region of interest (ROI) from the background
    rows, cols, _ = overlay_color.shape
    roi_bg = main_background[y_offset:y_offset + rows, x_offset:x_offset + cols]

    # Blend the PNG image with the ROI
    for c in range(0, 3):
        roi_bg[:, :, c] = roi_bg[:, :, c] * (1 - overlay_alpha) + overlay_color[:, :, c] * overlay_alpha

    # Place the blended region back into the original background image
    main_background[y_offset:y_offset + rows, x_offset:x_offset + cols] = roi_bg

    return main_background

# Function to detect the blue screen and replace it with a background frame
def replace_phone_screen(image, background_frame):
    global consecutive_frame_count
    global detected_for_required_period
    global start_zooming
    global zoom_scale

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame = image[:, :, :3]  # BGR channels
    overlay_color = image[:, :, :3]  # BGR channels
    overlay_alpha = image[:, :, 3]  # Alpha channel


    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    blurred_mask = cv2.GaussianBlur(blue_mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    blue_mask_cleaned = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(blue_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    frame[blue_mask != 0] = [0, 0, 0]

    if not contours:
        main_background = apply_background(overlay_alpha, overlay_color, background_frame)

        return main_background

    filtered_contours = [c for c in contours if cv2.contourArea(c) > 1700]  # Adjust the threshold as needed
    if not filtered_contours:
        main_background = apply_background(overlay_alpha, overlay_color, background_frame)
        return main_background

    largest_contour = max(filtered_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    background_phone_frame_resized = cv2.resize(background_frame, (w, h))

    roi = frame[y:y + h, x:x + w]

    phone_screen_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.drawContours(phone_screen_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(phone_screen_mask, [largest_contour], -1, (0, 0, 0), thickness=10, lineType=cv2.LINE_AA)

    phone_screen_mask_roi = phone_screen_mask[y:y + h, x:x + w]
    background_with_mask = cv2.bitwise_and(background_phone_frame_resized, background_phone_frame_resized,
                                           mask=phone_screen_mask_roi)
    inverse_mask = cv2.bitwise_not(phone_screen_mask_roi)
    roi_with_edges = cv2.bitwise_and(roi, roi, mask=inverse_mask)

    frame[y:y + h, x:x + w] = cv2.add(roi_with_edges, background_with_mask)

    # Update detection count
    consecutive_frame_count += 1
    # Check if the background frame is detected for more than 1 second

    ##
    main_background = apply_background(overlay_alpha, overlay_color, background_frame)
    ##

    # Apply zoom towards the center of the background in the main frame
    if start_zooming:
        zoom_scale += zoom_increment
        frame = apply_zoom_to_center(main_background, (x, y, w, h), background_frame)
        return frame

    if not start_zooming and consecutive_frame_count >= (required_frames_for_one_second):
        detected_for_required_period = True
        start_zooming = True
        print("Background detected for more than one second at frame:", int(video.get(cv2.CAP_PROP_POS_FRAMES)))

    return main_background


import cv2
import os

# Folder containing your PNG images
folder_path = './sequence/'
# The file naming pattern for the images (e.g., frame_001.png, frame_002.png, etc.)
file_pattern = '{:04d}.png'  # Modify this based on your file naming convention

# Loop to read images from 1 to 100 (for example)
for i in range(0, 206):
    file_name = file_pattern.format(i)
    file_path = os.path.join(folder_path, file_name)

    # Load the image
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Image {file_name} not found, skipping...")
        continue

    ret_bg, background_frame = background_video.read()
    processed_frame = replace_phone_screen(image, background_frame)

    output_video.write(processed_frame)

while background_video.isOpened():
    ret_bg, background_frame = background_video.read()
    if not ret_bg:
        break
    main_background = cv2.resize(background_frame, (frame_width, frame_height))
    output_video.write(main_background)

cv2.destroyAllWindows()


"""
# Process the video frame by frame
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    ret_pha, frame_pha = pha_video.read()
    if not ret_pha:
        break

    ret_bg, background_frame = background_video.read()
    if not ret_bg:
        background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_bg, background_frame = background_video.read()

    # if consecutive_frame_count and not start_zooming:

    # Apply zoom and replace the phone screen
    processed_frame = replace_phone_screen(frame, background_frame, frame_pha)

    output_video.write(processed_frame)
"""

# Release everything after the loop
video.release()
background_video.release()
output_video.release()
cv2.destroyAllWindows()
