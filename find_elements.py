import cv2
import base64
import numpy as np
from iterators.video_iterator import VideoIterator as VIter
from typing import List, Tuple, Dict, Any
from configs.elements_config import ElementsConfig
from classes.template import Template
from get_color_range import get_color

best_match = 0
best_val = 0
config = ElementsConfig()

def get_video(input_video_path: str, output_video_path: str, templates_list: List[Dict[str, Any]]):
    iterator = VIter(input_video_path)

    # Initialize templates as instances of Template class
    templates = get_templates(templates_list)
    # Get the width and height of the first frame for video settings
    first_frame = next(iterator)
    height, width, _ = first_frame.shape

    # Set up video recording (filename, codec, FPS, frame size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))

    for count_frames, frame in enumerate(iterator):
        print('Frame: ', count_frames)

        # Process frame only if the number of frames skipped is less than the current count
        if config.skip_frames <= count_frames:
            frame = elements_search(frame, templates)

        output_video.write(frame)

    output_video.release()


def get_frame_for_color(input_video_path: str):
    iterator = VIter(input_video_path)

    # Get the width and height of the first frame for video settings
    first_frame = next(iterator)

    get_color(first_frame)


def get_templates(templates_list: List[Dict[str, Any]]) -> List[Template]:
    """Creates a list of Template objects from given paths and settings."""
    templates = [Template(path=item['path'], resize=item['resize'], threshold=item['threshold']) for item in templates_list]
    return templates


def apply_zoom(frame: np.ndarray, center: Tuple[int, int], zoom_factor: float, width: int, height: int) -> np.ndarray:
    """Applies zoom to the frame around a specified center."""
    center_x, center_y = center
    new_w, new_h = int(width / zoom_factor), int(height / zoom_factor)

    # Position the zoom area based on the center
    new_top_left_x = max(0, center_x - new_w // 2)
    new_top_left_y = max(0, center_y - new_h // 2)
    new_bottom_right_x = min(new_top_left_x + new_w, width)
    new_bottom_right_y = min(new_top_left_y + new_h, height)

    # Crop and resize the frame to apply zoom
    roi_zoomed = frame[new_top_left_y:new_bottom_right_y, new_top_left_x:new_bottom_right_x]
    zoomed_frame = cv2.resize(roi_zoomed, (width, height))

    return zoomed_frame


def update_zoom(zoom_factor: float, zoom_direction: int, zoom_speed: float, max_zoom_factor: float) -> Tuple[float, int]:
    """Updates the zoom factor and its direction."""
    zoom_factor += zoom_direction * zoom_speed

    # Check for zoom limits
    if zoom_factor >= max_zoom_factor:
        zoom_factor = max_zoom_factor
        zoom_direction = -1
    elif zoom_factor <= 1.0:
        zoom_factor = 1.0
        zoom_direction = 1

    return zoom_factor, zoom_direction


def apply_darkening(frame: np.ndarray, template_gray: np.ndarray, scale: float, top_left: Tuple[int, int], darkness: float) -> np.ndarray:
    """Applies darkening around the area of the template."""
    if darkness <= 0:
        return frame

    # Create a darkened version of the frame based on the darkness level
    darkened_frame = cv2.addWeighted(frame, 1 - darkness, np.zeros_like(frame), darkness, 0)
    mask = np.ones_like(frame, dtype=np.uint8) * 255

    # Find contours and create a darkened area
    contours, _ = cv2.findContours(template_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        scaled_contour = contour * scale
        scaled_contour += np.array(top_left)
        cv2.drawContours(mask, [scaled_contour.astype(int)], -1, (0, 0, 0), thickness=cv2.FILLED)

    # Apply the mask to combine the original frame and the darkened frame
    frame = np.where(mask == 0, frame, darkened_frame)
    return frame


def update_darkness(darkness: float, zoom_direction: int, darkening_step: float) -> float:
    """Updates the level of darkness based on the zoom direction."""
    darkness += zoom_direction * darkening_step
    darkness = min(max(darkness, 0.0), 0.8)  # Clamp darkness between 0 and 0.8
    return darkness


def process_template(templates: List[Template], frame: np.ndarray, width: int, height: int, zoom_speed: float, max_zoom_factor: float):
    """Selects the current active template, updates its state, and returns data for processing."""
    # global best_match, best_val
    # Identify the first unfinished template
    active_template = next((template for template in templates if not template.completed), None)

    # If all templates are finished, return None
    if active_template is None:
        return None, None, None

    # Search for a match for the active template in the current frame
    find_best_match_full(frame, active_template)

    template_gray = cv2.cvtColor(cv2.imread(active_template.path), cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    print('best match', active_template.best_match)
    print('best val', active_template.best_val)

    # If a match is found, update its state
    if active_template.best_match and active_template.best_val >= active_template.threshold:
        print('Great!')
        top_left = active_template.best_match[0]
        scale = active_template.best_match[1]
        w_scaled, h_scaled = int(w * scale), int(h * scale)
        center = (top_left[0] + w_scaled // 2, top_left[1] + h_scaled // 2)

        # Update zoom and darkness based on current state
        active_template.zoom_factor, active_template.zoom_direction = update_zoom(
            active_template.zoom_factor, active_template.zoom_direction, zoom_speed, max_zoom_factor
        )
        darkening_step = 0.8 / ((max_zoom_factor - 1) / zoom_speed)
        active_template.darkness = update_darkness(active_template.darkness, active_template.zoom_direction, darkening_step)

        # Check if the template processing is complete
        if active_template.zoom_factor == 1.0 and active_template.zoom_direction == 1:
            active_template.completed = True  # Mark as completed if zoom is reset

        # Return relevant data for processing
        return active_template, (top_left, scale, center, active_template.darkness), active_template
    else:
        return active_template, None, active_template


def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def elements_search(frame: np.ndarray, templates: List[Template]) -> np.ndarray:
    """Main function for searching elements in the current frame using defined templates."""
    height, width, _ = frame.shape

    # Get active template and processing data
    active_template, processing_data, state = process_template(
        templates, frame, width, height, config.zoom_speed, config.max_zoom_factor
    )

    # If all templates are processed, return the original frame
    if active_template is None or processing_data is None:
        return frame

    # Process the frame with the current template
    top_left, scale, center, darkness = processing_data
    template_gray = cv2.cvtColor(cv2.imread(active_template.path), cv2.COLOR_BGR2GRAY)  # Convert template to grayscale
    frame = apply_darkening(frame, template_gray, scale, top_left, darkness)  # Apply darkening effect
    frame = apply_zoom(frame, center, active_template.zoom_factor, width, height)  # Apply zoom effect

    return frame


def increase_brightness(img, value=30):
    # Check if the image is grayscale
    if len(img.shape) == 2 or img.shape[2] == 1:
        # For grayscale, just add the brightness value directly
        img = cv2.add(img, np.array([value], dtype=np.uint8))
        img[img > 255] = 255  # Clip values to max 255
    elif img.shape[2] == 4:  # Check if the image has an alpha channel
        # Separate the RGB and alpha channels
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]

        # Convert the RGB channels to HSV
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Increase the brightness on the value channel
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        # Merge back the HSV channels and convert to BGR
        final_hsv = cv2.merge((h, s, v))
        bright_bgr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        # Combine the brightened BGR channels with the original alpha channel
        img = cv2.merge((bright_bgr, alpha))
    elif img.shape[2] == 3:
        # Process as a regular BGR image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    else:
        raise ValueError("Unsupported number of channels.")

    return img
def find_best_match_near_previous(image_gray: np.ndarray, template_gray: np.ndarray, best_match_info: Dict, search_window_size: int) -> Tuple:
    """Finds the best match for the template within a specified search window near the previous best match."""
    w, h = template_gray.shape[::-1]
    previous_x, previous_y = best_match_info["previous_best_match"][0]
    # Define search window around the previous match
    x_min = max(0, previous_x - search_window_size)
    x_max = min(image_gray.shape[1] - w, previous_x + search_window_size)
    y_min = max(0, previous_y - search_window_size)
    y_max = min(image_gray.shape[0] - h, previous_y + search_window_size)
    roi = image_gray[y_min:y_max, x_min:x_max]  # Region of Interest (ROI) for matching
    # Resize template based on previous scale
    resized_template = cv2.resize(template_gray, (int(w * best_match_info["previous_best_scale"]), int(h * best_match_info["previous_best_scale"])))
    result = cv2.matchTemplate(image_gray, resized_template, cv2.TM_CCORR)  # Template matching
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)  # Get match details
    return (max_loc, best_match_info["previous_best_scale"]), max_val  # Return match location and value

def convert_result_to_frame(frame, result):
    # Нормализуем result в диапазон 0-255
    result_norm = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result_norm = np.uint8(result_norm)  # Преобразуем к uint8 для совместимости

    # Преобразуем в цветное изображение, чтобы совпадало с каналами кадра (BGR)
    heatmap = cv2.applyColorMap(result_norm, cv2.COLORMAP_JET)

    # Накладываем тепловую карту на кадр
    overlay = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

    return overlay


def find_best_match_full(frame: np.ndarray, active_template: Template):
    """Finds the best match for a template across a range of scales based on specified min and max sizes."""
    # Retrieve template details
    template_gray = cv2.cvtColor(cv2.imread(active_template.path), cv2.COLOR_BGR2GRAY)

    last_scale = 0  # Initialize best match and value
    w, h = template_gray.shape[::-1]

    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert current frame to grayscale for processing
    image_gray = increase_brightness(image_gray)
    if active_template.best_match and active_template.best_val >= active_template.threshold:
        print('Popali')
        scales = [active_template.best_match[1]]
    else:
        print('Ne Popali')
        # Create scales based on the min and max sizes with a step of 5 pixels
        scales = np.arange(active_template.resize['min'] / min(w, h), active_template.resize['max'] / min(w, h) + 0.1, 0.05)  # Adjust step if needed

    for scale in scales:
        if last_scale:
            resized_template = cv2.resize(template_gray, (int(w * last_scale), int(h * last_scale)))
        else:
            resized_template = cv2.resize(template_gray, (int(w * scale), int(h * scale)))  # Resize template

        # Skip if resized template is larger than the image
        if resized_template.shape[0] > image_gray.shape[0] or resized_template.shape[1] > image_gray.shape[1]:
            last_scale = 0
            continue

        result = cv2.matchTemplate(image_gray, resized_template, cv2.TM_CCORR)  # Perform template matching
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)  # Get match details

        if max_val > active_template.best_val:  # Update best match if current one is better
            active_template.best_val = max_val
            active_template.best_match = (max_loc, scale)  # Store best match location and scale

    # If a best match is found, check the color on the original frame
    if active_template.best_match is not None:
        top_left = active_template.best_match[0]

        # Resize the template to the scale of the best match for masking purposes
        resized_template = cv2.resize(template_gray, (int(w * active_template.best_match[1]), int(h * active_template.best_match[1])))

        # Create a binary mask where the template is white (1) and the background is black (0)
        _, mask = cv2.threshold(resized_template, 1, 255, cv2.THRESH_BINARY)

        # Extract the region of interest (ROI) from the original frame
        roi = frame[top_left[1]:top_left[1] + resized_template.shape[0],
              top_left[0]:top_left[0] + resized_template.shape[1]]

        # Calculate the average color of the ROI using the mask to exclude the background
        avg_color = cv2.mean(roi, mask=mask.astype(np.uint8))[:3]  # Get average color (B, G, R) within the masked area


def is_color_within_tolerance(avg_color: Tuple[float, float, float], target_color: np.ndarray) -> bool:
    """Checks if the average color is within the specified tolerance of the target color."""
    for avg, target in zip(avg_color, target_color):
        lower_bound = max(0, target - config.tolerance)
        upper_bound = min(255, target + config.tolerance)
        if not (lower_bound <= avg <= upper_bound):
            return False  # Return false if any channel is out of bounds
    return True  # Return true if all channels are within bounds


def debug_image(img_path="link_img.jpg"):
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    frame = elements_search(
        image,
        [Template(path='./src/link/big-link.png', resize={'min': 120, 'max': 200}, color=np.array([200, 200, 200]))]
    )


def is_mostly_white(frame, sensitivity=90, threshold=0.8):
    # Define white color range in HSV
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])

    # Convert the frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for white pixels within the specified range
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)

    # Calculate the percentage of white pixels in the frame
    white_pixels = np.sum(white_mask == 255)
    total_pixels = frame.shape[0] * frame.shape[1]
    white_ratio = white_pixels / total_pixels

    # Check if the white ratio meets the threshold
    return white_ratio >= threshold


if __name__ == ('__main__'):
    # get_frame_for_color('temp/back_tiktok_temp.mp4')

    data = [
        # {'path': './src/share/big-share-white.png', 'resize': {'min': 120, 'max': 200}, 'threshold': 0.8},
        {'path': './src/link/link-white.jpg', 'resize': {'min': 120, 'max': 200}, 'threshold': 0.7}
    ]
    get_video(f'temp/phone_tiktok_temp.mp4', f'back_test_1.mp4', data)