import cv2
import numpy as np
import base64
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import moviepy.editor as mp

# Initialize the segmentation model
segmentor = SelfiSegmentation(model=0)

# Define the color ranges in HSV
lower_blue = np.array([80, 50, 80])
upper_blue = np.array([130, 255, 255])
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Detection tracking
consecutive_frame_count = 0
detected_for_required_period = False
start_zooming = False


# Function to apply saturation increase
def increase_saturation(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_frame)
    s = np.clip(cv2.multiply(s, 1.5), 0, 255).astype(np.uint8)
    hsv_adjusted = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)


# Function to extract green layers and modify saturation
def extract_green_layers(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_layer = cv2.bitwise_and(frame, frame, mask=green_mask)
    hsv[:, :, 1][green_mask != 0] = hsv[:, :, 1][green_mask != 0] * 0.1
    inverse_green_mask = cv2.bitwise_not(green_mask)

    hsv[:, :, 1][inverse_green_mask != 0] = np.clip(hsv[:, :, 1][inverse_green_mask != 0] * 1.4, 0, 255)

    modified_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return modified_frame, green_layer


# Function to adjust contrast and brightness
def adjust_contrast(frame, alpha, beta=0):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


# Convert a frame to base64
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


# Function to apply zoom towards the center of the background
def apply_zoom_to_center(main_frame, background_rect, background_frame, frame_width, frame_height, zoom_scale):
    h, w = main_frame.shape[:2]
    rect_x, rect_y, rect_w, rect_h = background_rect

    center_x, center_y = rect_x + rect_w // 2, rect_y + rect_h // 2
    new_w, new_h = int(w / zoom_scale), int(h / zoom_scale)

    if int(rect_w * zoom_scale) >= frame_width or int(rect_h * zoom_scale) >= frame_height:
        return background_frame

    x1, y1 = max(0, center_x - new_w // 2), max(0, center_y - new_h // 2)
    x2, y2 = min(w, center_x + new_w // 2), min(h, center_y + new_h // 2)

    cropped_frame = main_frame[y1:y2, x1:x2]
    return cv2.resize(cropped_frame, (w, h))


# Function to blend overlay and background
def apply_background(overlay_alpha, overlay_color, background_frame, frame_width, frame_height):
    main_background = cv2.resize(background_frame, (frame_width, frame_height))
    overlay_alpha = overlay_alpha / 255.0

    rows, cols, _ = overlay_color.shape
    roi_bg = main_background[:rows, :cols]

    for c in range(3):
        roi_bg[:, :, c] = roi_bg[:, :, c] * (1 - overlay_alpha) + overlay_color[:, :, c] * overlay_alpha

    main_background[:rows, :cols] = roi_bg
    return main_background


# Function to replace the phone screen with background
def replace_phone_screen(video, image, background_frame, required_frames_for_one_second, frame_width, frame_height, zoom_scale, zoom_increment):
    global consecutive_frame_count, detected_for_required_period, start_zooming

    frame = image[:, :, :3]
    overlay_color, overlay_alpha = image[:, :, :3], image[:, :, 3]

    # Extract green layers and process frame
    overlay_color, _ = extract_green_layers(overlay_color)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    blue_mask_cleaned = cv2.morphologyEx(cv2.GaussianBlur(blue_mask, (5, 5), 0), cv2.MORPH_CLOSE,
                                         np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(blue_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    frame[blue_mask != 0] = [0, 0, 0]

    if not contours:
        return apply_background(overlay_alpha, overlay_color, background_frame, frame_width, frame_height)

    filtered_contours = [c for c in contours if cv2.contourArea(c) > 1700]
    if not filtered_contours:
        return apply_background(overlay_alpha, overlay_color, background_frame, frame_width, frame_height)

    x, y, w, h = cv2.boundingRect(max(filtered_contours, key=cv2.contourArea))
    background_phone_frame_resized = cv2.resize(background_frame, (w, h))

    phone_screen_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(phone_screen_mask, [max(filtered_contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(phone_screen_mask, [max(filtered_contours, key=cv2.contourArea)], -1, (0, 0, 0), thickness=10,
                     lineType=cv2.LINE_AA)

    roi_with_edges = cv2.bitwise_and(frame[y:y + h, x:x + w], frame[y:y + h, x:x + w],
                                     mask=cv2.bitwise_not(phone_screen_mask[y:y + h, x:x + w]))
    frame[y:y + h, x:x + w] = cv2.add(roi_with_edges,
                                      cv2.bitwise_and(background_phone_frame_resized, background_phone_frame_resized,
                                                      mask=phone_screen_mask[y:y + h, x:x + w]))

    consecutive_frame_count += 1

    # Apply zoom using the passed zoom_scale and zoom_increment
    if start_zooming:
        zoom_scale += zoom_increment
        return apply_zoom_to_center(
            apply_background(overlay_alpha, overlay_color, background_frame, frame_width, frame_height), (x, y, w, h),
            background_frame, frame_width, frame_height, zoom_scale)

    if consecutive_frame_count >= required_frames_for_one_second and not start_zooming:
        detected_for_required_period = True
        start_zooming = True
        print("Background detected for more than one second.")

    return apply_background(overlay_alpha, overlay_color, background_frame, frame_width, frame_height)


# Main chroma replace function
def chroma_replace(video_path: str, full_background_path: str, phone_background_path: str, zoom_scale: float, zoom_increment: float):
    video = cv2.VideoCapture(video_path)
    background_video = mp.VideoFileClip(full_background_path).set_fps(int(video.get(cv2.CAP_PROP_FPS)))

    if not os.path.exists('temp'):
        os.makedirs('temp')

    temp_full_back_path = os.path.join('temp',
                                       f'{os.path.splitext(os.path.basename(full_background_path))[0]}_temp.mp4')
    background_video.write_videofile(temp_full_back_path, codec='libx264')
    background_video = cv2.VideoCapture(temp_full_back_path)

    frame_width, frame_height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    output_video = cv2.VideoWriter(f'{os.path.splitext(video_path)[0]}_detected_video.mp4',
                                   cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    required_frames_for_one_second = fps

    folder_path = f'./sequence/{os.path.splitext(video_path)[0]}'
    file_pattern = '{:04d}.png'
    num_files = len([f for f in os.listdir(folder_path) if f.endswith('.png')])

    for i in range(num_files):
        file_name = file_pattern.format(i)
        file_path = os.path.join(folder_path, file_name)
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(f"Image {file_name} not found, skipping...")
            continue

        ret_bg, background_frame = background_video.read()
        processed_frame = replace_phone_screen(video, image, background_frame, required_frames_for_one_second,
                                               frame_width, frame_height, zoom_scale, zoom_increment)

        output_video.write(processed_frame)

    cv2.destroyAllWindows()
    video.release()
    background_video.release()
    output_video.release()