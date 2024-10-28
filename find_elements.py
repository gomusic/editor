import cv2
import base64
import numpy as np
from iterators.video_iterator import VideoIterator as VIter
from typing import List, Tuple, Dict, Any
from elements_config import ElementsConfig
from classes.template import Template


config = ElementsConfig()

def get_video(input_video_path: str, output_video_path: str, templates_list: List[Dict[str, Any]]):
    iterator = VIter(input_video_path)

    # Get the width and height of the first frame for video settings
    first_frame = next(iterator)
    height, width, _ = first_frame.shape

    # Set up video recording (filename, codec, FPS, frame size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))

    # Initialize templates as instances of Template class
    templates = get_templates(templates_list)

    for count_frames, frame in enumerate(iterator):
        print('Frame: ', count_frames)

        # Process frame only if the number of frames skipped is less than the current count
        if config.skip_frames <= count_frames:
            frame = elements_search(frame, templates)

        output_video.write(frame)

    output_video.release()


def get_templates(templates_list: List[Dict[str, Any]]) -> List[Template]:
    """Creates a list of Template objects from given paths and settings."""
    templates = [Template(path=item['path'], resize=item['resize'], colors=item['colors']) for item in templates_list]
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


def process_template(templates: List[Template], image_gray: np.ndarray, width: int, height: int, zoom_speed: float, max_zoom_factor: float):
    """Selects the current active template, updates its state, and returns data for processing."""

    # Identify the first unfinished template
    active_template = next((template for template in templates if not template.completed), None)

    # If all templates are finished, return None
    if active_template is None:
        return None, None, None

    # Retrieve template details
    template_gray = cv2.cvtColor(cv2.imread(active_template.path), cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    # Search for a match for the active template in the current frame
    best_match, best_val = find_best_match_full(image_gray, template_gray, min_size=active_template.resize['min'], max_size=active_template.resize['max'])

    print('best match', best_match)
    print('best val', best_val)

    # If a match is found, update its state
    if best_match and best_val >= config.threshold:
        print('Great!')
        top_left = best_match[0]
        scale = best_match[1]
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
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert current frame to grayscale for processing

    # Get active template and processing data
    active_template, processing_data, state = process_template(
        templates, image_gray, width, height, config.zoom_speed, config.max_zoom_factor
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
    result = cv2.matchTemplate(roi, resized_template, cv2.TM_CCOEFF_NORMED)  # Template matching
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)  # Get match details
    return (max_loc, best_match_info["previous_best_scale"]), max_val  # Return match location and value


def find_best_match_full(image_gray: np.ndarray, template_gray: np.ndarray, min_size: int, max_size: int) -> Tuple:
    """Finds the best match for a template across a range of scales based on specified min and max sizes."""
    best_match, best_val = None, 0  # Initialize best match and value
    w, h = template_gray.shape[::-1]

    # Create scales based on the min and max sizes with a step of 5 pixels
    scales = np.arange(min_size / min(w, h), max_size / min(w, h) + 0.1, 0.05)  # Adjust step if needed

    for scale in scales:
        resized_template = cv2.resize(template_gray, (int(w * scale), int(h * scale)))  # Resize template

        # Skip if resized template is larger than the image
        if resized_template.shape[0] > image_gray.shape[0] or resized_template.shape[1] > image_gray.shape[1]:
            continue

        result = cv2.matchTemplate(image_gray, resized_template, cv2.TM_CCOEFF_NORMED)  # Perform template matching
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)  # Get match details

        if max_val > best_val:  # Update best match if current one is better
            best_val = max_val
            best_match = (max_loc, scale)  # Store best match location and scale

    return best_match, best_val  # Return best match and value


if __name__ == ('__main__'):
    data = [
        {'path': './src/share/big-share.png', 'resize': {'min': 20, 'max': 200}, 'colors': np.array([255, 255, 255])},
        {'path': './src/comment/big-comment.png', 'resize': {'min': 20, 'max': 200}, 'colors': np.array([255, 255, 255])},
        {'path': './src/link/big-link-test.png', 'resize': {'min': 20, 'max': 200}, 'colors': np.array([45, 100, 242])}
    ]
    get_video(f'elements.mp4', f'test3.mp4', data)
    pass
