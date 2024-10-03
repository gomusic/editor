import cv2
import numpy as np


def increase_saturation(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Split the channels: H (Hue), S (Saturation), and V (Value)
    h, s, v = cv2.split(hsv_frame)

    # Adjust the saturation by multiplying with a factor (1.0 means no change)
    saturation_scale = 2  # Increase saturation by 50%
    s = cv2.multiply(s, saturation_scale)

    # Clip the values to ensure they are in the valid range [0, 255]
    s = np.clip(s, 0, 255).astype(np.uint8)

    # Merge the channels back
    hsv_adjusted = cv2.merge([h, s, v])

    # Convert the image back to BGR color space
    corrected_frame = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

    return corrected_frame
def correct_color(frame):
    # Convert the image to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split into channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to L-channel (lightness)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # Merge CLAHE enhanced L-channel with a and b channels
    limg = cv2.merge((cl, a_channel, b_channel))

    # Convert LAB back to BGR
    corrected_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return corrected_frame


def correct_white_balance(frame):
    # Convert to gray and compute gray world white balance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)

    # Scale each channel based on mean intensity to achieve gray world assumption
    b_mean, g_mean, r_mean = np.mean(frame, axis=(0, 1))
    b_scale = mean_intensity / b_mean
    g_scale = mean_intensity / g_mean
    r_scale = mean_intensity / r_mean

    # Apply scaling factors to each channel
    frame[:, :, 0] = np.clip(frame[:, :, 0] * b_scale, 0, 255)  # Blue channel
    frame[:, :, 1] = np.clip(frame[:, :, 1] * g_scale, 0, 255)  # Green channel
    frame[:, :, 2] = np.clip(frame[:, :, 2] * r_scale, 0, 255)  # Red channel

    return frame


video = cv2.VideoCapture('headphones.mp4')

# Get the properties of the main video
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

"""
Clicked pixel BGR: [168 162 136]
Clicked pixel HSV: [ 96  49 168]
"""

center_x = frame_width // 2
center_y = frame_height // 2
colors = [
    { 'old_color': [160, 0, 100], 'new_color': [200, 150, 150] },
    #{ 'old_color': [104, 136, 110], 'new_color': [14, 63, 69]},
]


green_lower = np.array([35, 40, 40])
green_upper = np.array([85, 255, 255])
radius = 20  # Define the radius to check around the contour



while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define a mask range that well captures your object
    mask = cv2.inRange(hsv, np.array([80,0,140]), np.array([130, 100, 180]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    square_contours = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        M = cv2.moments(cnt)
        #if cv2.contourArea(cnt) < 700:
        #    continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Expand bounding box by radius
        x_start = max(0, x - radius)
        y_start = max(0, y - radius)
        x_end = min(hsv.shape[1], x + w + radius)
        y_end = min(hsv.shape[0], y + h + radius)

        # Extract the background area around the contour
        background_area = hsv[y_start:y_end, x_start:x_end]

        # Check how much of the background is green
        background_mask = cv2.inRange(background_area, green_lower, green_upper)
        green_pixels = cv2.countNonZero(background_mask)
        total_pixels = background_area.shape[0] * background_area.shape[1]

        green_ratio = green_pixels / total_pixels

        if green_ratio > 0.3:  # You can adjust this threshold based on how "green" you want the background to be
            square_contours.append(approx)

        #if cv2.contourArea(cnt) < 600:
        #    continue
        # square_contours.append(approx)

    # Draw or mask each square-like contour
    for contour in square_contours:
        cv2.drawContours(frame, [contour], -1, (110, 141, 116), thickness=cv2.FILLED)  # Draw green contours
        cv2.drawContours(frame, [contour], -1, (110, 141, 116), thickness=10, lineType=cv2.LINE_AA)  # Draw green contours

    frame = increase_saturation(frame)
    #frame = correct_white_balance(frame)
    # Write the frame to the output file
    output_video.write(frame)

    # Optionally display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break


"""
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    for color in colors:
        lower = color['old_color']
        lower[0] = lower[0] - 40
        lower = np.array(lower)

        upper = color['old_color']
        upper[0] = upper[0] + 40
        upper = np.array(upper)

        #mask = cv2.inRange(hsv, np.array([90, 50, 70]), np.array([133, 158, 255]))
        mask = cv2.inRange(hsv, np.array([80,0,140]), np.array([130, 100, 180]))


        # этот диапазон хорошо захватывает человека
        #mask = cv2.inRange(hsv, np.array([0,20,0]), np.array([50,255,255]))
        #cv2.imshow('Frame', mask)
        new_color = color['new_color']  # BGR for red
        frame[mask > 0] = [0,0,0]

    # Write the frame to the output file
    output_video.write(frame)

    # Optionally display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        """

video.release()
output_video.release()
cv2.destroyAllWindows()
