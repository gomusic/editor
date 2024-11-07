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
    templates = [Template(**item) for item in templates_list]
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

    template_gray = cv2.cvtColor(cv2.imread(active_template.template_path), cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    print('best match', active_template.best_match)
    print('best val', active_template.best_val)

    # If a match is found, update its state
    if active_template.best_match and active_template.best_val >= active_template.threshold:
        print(active_template.threshold)
        print(active_template.best_val)
        print(active_template.best_match)
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
    frame = cv2.fastNlMeansDenoisingColored(frame)

    # Get active template and processing data
    active_template, processing_data, state = process_template(
        templates, frame, width, height, config.zoom_speed, config.max_zoom_factor
    )

    # If all templates are processed, return the original frame
    if active_template is None or processing_data is None:
        return frame

    # Process the frame with the current template
    top_left, scale, center, darkness = processing_data
    template_gray = cv2.cvtColor(cv2.imread(active_template.template_path), cv2.COLOR_BGR2GRAY)
    frame = apply_darkening(frame, template_gray, scale, top_left, darkness)  # Apply darkening effect
    frame = apply_zoom(frame, center, active_template.zoom_factor, width, height)  # Apply zoom effect

    return frame



def draw_circle(frame: np.ndarray, center: tuple, scale: float, form_size: int, template_width: int, template_height: int) -> np.ndarray:
    """Draws a circle on the frame based on the provided center and size parameters."""
    # Вычисляем базовый радиус
    base_radius = int(0.5 * ((template_width * scale) ** 2 + (template_height * scale) ** 2) ** 0.5)
    radius = base_radius + form_size

    # Задаем цвет и толщину линии круга
    color = (0, 255, 0)  # Зеленый цвет в формате BGR
    thickness = 2  # Толщина обводки

    # Рисуем круг с центром в center и радиусом radius
    cv2.circle(frame, center, radius, color, thickness)

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

    if active_template.background_hex_color:
        lower, upper, l2, u2 = hex_to_hsv_range(active_template.background_hex_color)
        mask = get_hsv_mask(frame, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    template_gray = cv2.cvtColor(cv2.imread(active_template.template_path), cv2.COLOR_BGR2GRAY)
    template_gray = cv2.fastNlMeansDenoising(template_gray)
    w, h = template_gray.shape[::-1]

    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.fastNlMeansDenoising(image_gray)

    if active_template.best_match and active_template.best_val >= active_template.threshold:
        scales = [active_template.best_match[1]]
    else:
        scales = np.arange(
            active_template.resize['min'] / min(w, h),
            active_template.resize['max'] / min(w, h) + 0.1,
            0.05
        )

    best_match = None
    best_score = 0

    for scale in scales:
        resized_template = cv2.resize(template_gray, (int(w * scale), int(h * scale)))
        if resized_template.shape[0] > image_gray.shape[0] or resized_template.shape[1] > image_gray.shape[1]:
            continue

        if active_template.background_hex_color:
            for contour in contours:
                x, y, w_contour, h_contour = cv2.boundingRect(contour)
                roi = image_gray[y:y + h_contour, x:x + w_contour]

                if roi.shape[0] < resized_template.shape[0] or roi.shape[1] < resized_template.shape[1]:
                    continue

                result = cv2.matchTemplate(roi, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > active_template.best_val:
                    active_template.best_val = max_val
                    active_template.best_match = ((x + max_loc[0], y + max_loc[1]), scale)

                    _, binary_mask = cv2.threshold(resized_template, 1, 255, cv2.THRESH_BINARY)

                    resized_binary_mask = cv2.resize(binary_mask, (roi.shape[1], roi.shape[0]),
                                                     interpolation=cv2.INTER_NEAREST)

                    roi_mask = mask[y:y + h_contour, x:x + w_contour]
                    match_score = np.sum(cv2.bitwise_and(roi_mask, resized_binary_mask) == 255)

                    if match_score > best_score:
                        best_score = match_score
                        best_match = ((x + max_loc[0], y + max_loc[1]), scale)
        else:
            result = cv2.matchTemplate(image_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > active_template.best_val:
                active_template.best_val = max_val
                active_template.best_match = (max_loc, scale)

    if best_match:
        active_template.best_match = best_match


def is_color_within_tolerance(avg_color: Tuple[float, float, float], target_color: np.ndarray) -> bool:
    """Checks if the average color is within the specified tolerance of the target color."""
    for avg, target in zip(avg_color, target_color):
        lower_bound = max(0, target - config.tolerance)
        upper_bound = min(255, target + config.tolerance)
        if not (lower_bound <= avg <= upper_bound):
            return False  # Return false if any channel is out of bounds
    return True  # Return true if all channels are within bounds


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


def hex_to_hsv_range(hex_color, hue_tol=10, sat_tol=40, val_tol=40):
    # Convert HEX to BGR
    hex_color = hex_color.lstrip('#')
    bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

    # Create a single pixel array with the BGR color
    bgr_pixel = np.uint8([[bgr]])

    # Convert BGR to HSV
    hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)
    hsv_value = hsv_pixel[0][0]

    # Define ranges with specified tolerance
    lower_hue = (hsv_value[0] - hue_tol) % 180
    upper_hue = (hsv_value[0] + hue_tol) % 180
    lower_sat = max(hsv_value[1] - sat_tol, 0)
    upper_sat = min(hsv_value[1] + sat_tol, 255)
    lower_val = max(hsv_value[2] - val_tol, 0)
    upper_val = min(hsv_value[2] + val_tol, 255)

    # Adjust hue wrap-around issue
    if lower_hue > upper_hue:
        lower_range1 = (0, lower_sat, lower_val)
        upper_range1 = (upper_hue, upper_sat, upper_val)
        lower_range2 = (lower_hue, lower_sat, lower_val)
        upper_range2 = (179, upper_sat, upper_val)
        return np.array(lower_range1), np.array(upper_range1), np.array(lower_range2), np.array(upper_range2)
    else:
        lower_range = (lower_hue, lower_sat, lower_val)
        upper_range = (upper_hue, upper_sat, upper_val)
        return np.array(lower_range), np.array(upper_range), None, None


def contour_have_black_pixels_neibs(contour, mask):
    try:
        x, y = contour[0][0]
        if mask[y - 1, x] == 0:
            return True

        return False
    except:
        return True


def get_hsv_mask(frame, lower_hsv, upper_hsv, is_white=False):
    if is_white:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    image = cv2.fastNlMeansDenoisingColored(frame)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if not contour_have_black_pixels_neibs(c, mask)]
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    return mask


def display_hsv_highlight(image_path, lower_hsv, upper_hsv, is_white = False):
    # Load the original image
    image = cv2.imread(image_path)
    #image = cv2.convertScaleAbs(image, alpha=1, beta=1.5)
    if image is None:
        print("Error: Image not loaded. Check the file path.")
        return

    # Convert the image from BGR to HSV color space
    if is_white:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


    image = cv2.fastNlMeansDenoisingColored(image) # Reducing image noise
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for the specified HSV range
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    # Create an output image that only shows the masked area in color
    result_image = cv2.bitwise_and(image, image, mask=mask)

    # Optionally, you can overlay the mask on the original image to highlight the detected areas
    # Convert the mask to a three-channel image
    colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    highlighted_image = cv2.addWeighted(image, 1, colored_mask, 0.25, 0)  # Adjust transparency as needed

    # Display the original, mask, and result images
    #cv2.imshow('Original Image', image)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if not contour_have_black_pixels_neibs(c, mask)]


    cv2.drawContours(result_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    debug_image(image=result_image)
    cv2.imshow('Mask', result_image)
    #cv2.imshow('Mask Applied to Image', result_image)
    #cv2.imshow('Highlighted Image', highlighted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def hex_to_hsv(hex_color):
    # Step 1: Hex to BGR
    # Convert hex to an RGB tuple
    hex_color = hex_color.lstrip('#')  # Remove the '#' symbol if it's there
    bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))  # Convert RGB to BGR

    # Create an array containing the BGR value (as a single pixel)
    bgr_pixel = np.uint8([[bgr]])

    # Step 2: BGR to HSV
    hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)

    return hsv_pixel[0][0]


def debug_image(image = None, image_path = None):
    # threshold = 0.5
    # # Загрузим изображение по пути
    if image_path:
        image = cv2.imread(image_path)
    #     image = cv2.fastNlMeansDenoisingColored(image)  # Reducing image noise
    #
    # if image is None:
    #     print("Ошибка: Не удалось загрузить изображение.")
    #     return
    #
    # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # template = cv2.imread('./src/link/tiktok_link.png')
    #
    # if template is None:
    #     print("Ошибка: Не удалось загрузить шаблонное изображение.")
    #     return
    #
    # template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # h, w, channels = image.shape
    # # img_gray = cv2.fastNlMeansDenoising(img_gray)
    # # template_gray = cv2.fastNlMeansDenoising(template_gray)
    # res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    #
    # loc = np.where(res >= threshold)
    # th, tw = template_gray.shape[:2]
    #
    # for pt in zip(*loc[::-1]):  # Поменяем x и y координаты
    #     top_left = pt
    #     bottom_right = (top_left[0] + tw, top_left[1] + th)
    #     cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    #
    # cv2.imwrite('res.jpg', image)
    # return

    frame = elements_search(
        image,
        [Template(template_path='src/link/tiktok_link.png', resize={'min': 15, 'max': 20}, threshold=0.6, background_hex_color='#2764FB')]
    )

    cv2.imwrite('res2.jpg', frame)

if __name__ == ('__main__'):
    # lower, upper, lower2, upper2 = hex_to_hsv_range('#ffffff', hue_tol=0, sat_tol=0, val_tol=30)
    # lower, upper, l2, u2 = hex_to_hsv_range('#2764FB')
    # image = cv2.imread("new_tests/img_4.png")
    # image = get_hsv_mask(image, lower, upper)
    # cv2.imwrite('res2.jpg', image)
    # hsv_lower = np.array([24, 200, 255])
    # display_hsv_highlight("new_tests/img_4.png", lower, upper)
    # debug_image(image_path="new_tests/tik_img_test.jpg")
    # exit(0)
    #get_frame_for_color('temp/back_tiktok_temp.mp4')
    data = [
        {'template_path': './src/share/big-share-white.png', 'resize': {'min': 80, 'max': 120}, 'threshold': 0.8},
        {'template_path': './src/link/tiktok_link.png', 'resize': {'min': 15, 'max': 20}, 'threshold': 0.6, 'background_hex_color': '#2764FB'}
    ]
    get_video(f'temp/phone_tiktok_temp.mp4', f'back_test_1.mp4', data)