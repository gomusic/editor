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

            # Get the minimum-area bounding rectangle (which can be rotated)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Define the source points (corners of the background video)
            h, w = background_frame.shape[:2]
            pts_src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

            # Compute the homography matrix to warp the background video to the detected area
            h_matrix, _ = cv2.findHomography(pts_src, box.astype(np.float32))
            warped_background = cv2.warpPerspective(background_frame, h_matrix, (frame.shape[1], frame.shape[0]))

            # Create a mask for the phone screen area
            phone_screen_mask = np.zeros_like(frame[:, :, 0])
            cv2.drawContours(phone_screen_mask, [box], -1, 255, thickness=cv2.FILLED)

            # Apply the mask to the warped background to keep it within the phone screen area
            warped_background_masked = cv2.bitwise_and(warped_background, warped_background, mask=phone_screen_mask)

            # Create an inverse mask to keep the area around the phone screen intact
            inverse_mask = cv2.bitwise_not(phone_screen_mask)
            frame_with_edges = cv2.bitwise_and(frame, frame, mask=inverse_mask)

            # Combine the preserved areas with the overlaid background video on the phone screen
            combined_frame = cv2.add(frame_with_edges, warped_background_masked)

            return combined_frame

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
