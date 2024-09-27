import cv2
import torch
import numpy as np
import yolov5

# Load the YOLO model (YOLOv5s, pre-trained on COCO dataset)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = yolov5.load('yolov5s.pt')

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

# Prepare optical flow parameters (for smooth transitions)
prev_gray = None

# Function to find homography and apply perspective transform
def overlay_transformed_background(frame, phone_corners, background_frame):
    # Define destination points (corners of the phone screen)
    dst_points = np.float32([
        [0, 0],
        [background_frame.shape[1], 0],
        [background_frame.shape[1], background_frame.shape[0]],
        [0, background_frame.shape[0]]
    ])

    # Resize background frame to match the phone screen's bounding box
    background_frame_resized = cv2.resize(background_frame, (int(phone_corners[2][0] - phone_corners[0][0]), int(phone_corners[2][1] - phone_corners[0][1])))

    # Find the homography matrix
    M, _ = cv2.findHomography(dst_points, phone_corners)

    # Warp the background frame to fit the phone screen
    transformed_bg = cv2.warpPerspective(background_frame_resized, M, (frame.shape[1], frame.shape[0]))

    # Create a mask to only apply the background to the phone screen area
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(phone_corners)], (255, 255, 255))

    # Inverse the mask
    inverse_mask = cv2.bitwise_not(mask)

    # Keep the rest of the frame intact
    original_frame = cv2.bitwise_and(frame, inverse_mask)

    # Combine the transformed background with the original frame
    result_frame = cv2.add(original_frame, transformed_bg)

    return result_frame

# Main loop to process the video frame by frame
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Use YOLO to detect objects in the frame
    results = model(frame)

    # Filter for 'cell phone' class (class_id = 67 in COCO)
    detected_phones = results.xyxy[0][results.xyxy[0][:, 5] == 67]  # YOLO outputs boxes in [x1, y1, x2, y2, conf, class]

    if len(detected_phones) > 0:
        # Get the bounding box of the detected phone
        x1, y1, x2, y2, _, _ = detected_phones[0]

        # Define the phone corners (for homography)
        phone_corners = np.float32([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ])

        # Read a frame from the background video
        ret_bg, background_frame = background_video.read()

        # If the background video ends, restart it
        if not ret_bg:
            background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_bg, background_frame = background_video.read()

        # Apply homography and overlay the background video onto the phone screen
        processed_frame = overlay_transformed_background(frame, phone_corners, background_frame)

        # Write the processed frame to the output video
        output_video.write(processed_frame)

        # Show the result in real time (optional)
        cv2.imshow("Processed Frame", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        # Write the unprocessed frame (if no phone is detected)
        output_video.write(frame)

# Release everything after the loop
video.release()
background_video.release()
output_video.release()
cv2.destroyAllWindows()