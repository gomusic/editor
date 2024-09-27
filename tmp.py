import cv2
import numpy as np

# Load the main video
video = cv2.VideoCapture('correction.mp4')

# Load the background video that will replace the phone screen
background_video = cv2.VideoCapture('rotated.mov')

# Get the properties of the main video
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output_detected_video.mp4', fourcc, fps, (frame_width, frame_height))

# Define the blue color range in HSV for detecting the phone screen
lower_blue = np.array([90, 100, 100])  # adjust based on your blue color range
upper_blue = np.array([130, 255, 255])  # adjust based on your blue color range

# Function to detect the blue screen, ignore inner dots, and replace it with a background frame
def replace_phone_screen(frame, background_frame):
    # Convert the frame to HSV color space for blue detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for blue color (phone screen)
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Find contours in the blue mask
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If any contours are detected
    if contours:
        # Filter out smaller contours (dots)
        filtered_contours = [c for c in contours if cv2.contourArea(c) > 500]  # Adjust the threshold as needed

        if filtered_contours:
            # Find the largest contour which corresponds to the main blue area
            largest_contour = max(filtered_contours, key=cv2.contourArea)

            # Get the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Resize the background frame to the size of the detected phone screen area
            background_frame_resized = cv2.resize(background_frame, (w, h))

            # Extract the region of interest (ROI) where the phone screen is detected
            roi = frame[y:y+h, x:x+w]

            # Create a clean mask only for the largest contour (phone screen), ensure it's single-channel
            phone_screen_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.drawContours(phone_screen_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            # Extract the mask for the ROI (the detected screen area)
            phone_screen_mask_roi = phone_screen_mask[y:y+h, x:x+w]

            # Ensure the background video fits only inside the phone screen using the mask
            background_with_mask = cv2.bitwise_and(background_frame_resized, background_frame_resized, mask=phone_screen_mask_roi)

            # Keep the surrounding area of the phone intact by using an inverse mask
            inverse_mask = cv2.bitwise_not(phone_screen_mask_roi)
            roi_with_edges = cv2.bitwise_and(roi, roi, mask=inverse_mask)

            # Combine the preserved edges with the replaced screen
            frame[y:y+h, x:x+w] = cv2.add(roi_with_edges, background_with_mask)

    return frame

# Process the video frame by frame
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Read a frame from the background video
    ret_bg, background_frame = background_video.read()

    # If the background video ends, restart it (optional)
    if not ret_bg:
        background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_bg, background_frame = background_video.read()

    # Replace the detected blue screen (phone screen) with the background video frame
    processed_frame = replace_phone_screen(frame, background_frame)

    # Write the processed frame to the output video
    output_video.write(processed_frame)

# Release everything after the loop
video.release()
background_video.release()
output_video.release()
cv2.destroyAllWindows()
