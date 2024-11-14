import cv2
import numpy as np


# Function to increase saturation
def increase_saturation(frame, saturation_scale=2):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_frame)
    s = cv2.multiply(s, saturation_scale)
    s = np.clip(s, 0, 255).astype(np.uint8)
    hsv_adjusted = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)


# Function to apply CLAHE for color correction
def correct_color(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


# Function to correct white balance using Gray World Assumption
def correct_white_balance(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    b_mean, g_mean, r_mean = np.mean(frame, axis=(0, 1))
    scales = [mean_intensity / b_mean, mean_intensity / g_mean, mean_intensity / r_mean]

    for i, scale in enumerate(scales):
        frame[:, :, i] = np.clip(frame[:, :, i] * scale, 0, 255)

    return frame


# Function to check if a background area has a significant amount of green
def is_green_background(hsv_area, green_lower, green_upper, green_threshold=0.3):
    background_mask = cv2.inRange(hsv_area, green_lower, green_upper)
    green_pixels = cv2.countNonZero(background_mask)
    total_pixels = hsv_area.shape[0] * hsv_area.shape[1]
    return (green_pixels / total_pixels) > green_threshold


# Function to process each frame and replace specific colors
def process_frame(frame, green_lower, green_upper, radius=20):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([80, 0, 140]), np.array([130, 100, 180]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    square_contours = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x, y, w, h = cv2.boundingRect(cnt)
        x_start, y_start = max(0, x - radius), max(0, y - radius)
        x_end, y_end = min(hsv.shape[1], x + w + radius), min(hsv.shape[0], y + h + radius)
        background_area = hsv[y_start:y_end, x_start:x_end]

        if is_green_background(background_area, green_lower, green_upper):
            square_contours.append(approx)

    for contour in square_contours:
        cv2.drawContours(frame, [contour], -1, (110, 141, 116), thickness=cv2.FILLED)  # Draw filled green contours
        cv2.drawContours(frame, [contour], -1, (110, 141, 116), thickness=10, lineType=cv2.LINE_AA)  # Outline contours

    return increase_saturation(frame)


# Main function to process the video
def replace(editor_config):
    video = cv2.VideoCapture(editor_config.original_video)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(editor_config.replace_output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video.read()
        if not ret:
            break

        processed_frame = process_frame(frame, editor_config.lower_green, editor_config.upper_green)
        output_video.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    output_video.release()
    cv2.destroyAllWindows()