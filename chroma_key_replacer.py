import cv2
import numpy as np
import base64
import os
import moviepy.editor as mp
import mediapipe
from configs.editor_config import EditorConfig
from iterators.png_iterator import FrameIterator as FIter
from iterators.video_iterator import VideoIterator as VIter
from PIL import Image, ImageFilter


# Initialize the segmentation model
mediapipe_selfie_segmentation = mediapipe.solutions.selfie_segmentation
mediapipe_segmentor = mediapipe_selfie_segmentation.SelfieSegmentation(model_selection=1)
start_phone_video = False
global_editor_config = EditorConfig()

# Detection tracking
consecutive_frame_count = 0
detected_for_required_period = False
start_zooming = False
global_zoom_scale = 0.2

# Function to apply saturation increase
def increase_saturation(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_frame)
    s = np.clip(cv2.multiply(s, 1.5), 0, 255).astype(np.uint8)
    hsv_adjusted = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)


# Function to extract green layers and modify saturation
def extract_green_layers(frame, lower_green, upper_green):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Reduce saturation in green areas
    hsv[:, :, 1][green_mask != 0] = hsv[:, :, 1][green_mask != 0] * 0.1

    # Increase saturation in non-green areas
    inverse_green_mask = cv2.bitwise_not(green_mask)
    hsv[:, :, 1][inverse_green_mask != 0] = np.clip(hsv[:, :, 1][inverse_green_mask != 0] * 1.4, 0, 255)

    modified_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return modified_frame


# Function to adjust contrast and brightness
def adjust_contrast(frame, alpha, beta=0):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

# Convert a frame to base64
def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


# Function to apply zoom towards the center of the background
def apply_zoom_to_center(main_frame, background_rect, background_frame, phone_frame, frame_width, frame_height, zoom_scale):
    global start_phone_video
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
        start_phone_video = True
        return phone_frame

    # Calculate the coordinates for cropping the zoomed-in frame
    x1 = max(0, center_x - new_w // 2)
    y1 = max(0, center_y - new_h // 2)
    x2 = min(w, center_x + new_w // 2)
    y2 = min(h, center_y + new_h // 2)

    cropped_frame = main_frame[y1:y2, x1:x2]

    # Resize back to the original size (zoom effect)
    zoomed_frame = cv2.resize(cropped_frame, (w, h))

    return zoomed_frame


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
def replace_phone_screen_png(image, background_frame, phone_frame, required_frames_for_one_second, frame_width, frame_height):
    global consecutive_frame_count
    global detected_for_required_period
    global start_zooming
    global global_editor_config

    w = frame_width
    h = frame_height

    frame = image[:, :, :3]
    overlay_alpha = image[:, :, 3]
    overlay_color = frame

    # Convert frame to HSV and apply blue mask
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv_frame, global_editor_config.lower_blue, global_editor_config.upper_blue)
    blurred_mask = cv2.GaussianBlur(blue_mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    blue_mask_cleaned = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(blue_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    frame[blue_mask != 0] = [0, 0, 0]

    if not contours:
        main_background = apply_background(overlay_alpha, overlay_color, background_frame, frame_width, frame_height)
        main_background = extract_green_layers(main_background, global_editor_config.lower_green, global_editor_config.upper_green)
        return main_background

    filtered_contours = [c for c in contours if cv2.contourArea(c) > 1700]
    if not filtered_contours:
        main_background = apply_background(overlay_alpha, overlay_color, background_frame, frame_width, frame_height)
        main_background = extract_green_layers(main_background, global_editor_config.lower_green, global_editor_config.upper_green)
        return main_background

    largest_contour = max(filtered_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    background_phone_frame_resized = cv2.resize(phone_frame, (w, h))
    roi = frame[y:y + h, x:x + w]

    phone_screen_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv2.drawContours(phone_screen_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(phone_screen_mask, [largest_contour], -1, (0, 0, 0), thickness=10, lineType=cv2.LINE_AA)

    phone_screen_mask_roi = phone_screen_mask[y:y + h, x:x + w]
    background_with_mask = cv2.bitwise_and(background_phone_frame_resized, background_phone_frame_resized, mask=phone_screen_mask_roi)
    inverse_mask = cv2.bitwise_not(phone_screen_mask_roi)
    roi_with_edges = cv2.bitwise_and(roi, roi, mask=inverse_mask)
    frame[y:y + h, x:x + w] = cv2.add(roi_with_edges, background_with_mask)

    consecutive_frame_count += 1
    main_background = apply_background(overlay_alpha, overlay_color, background_frame, frame_width, frame_height)

    # Apply green layer extraction only outside the phone frame area
    phone_frame_mask_inv = cv2.bitwise_not(phone_screen_mask)  # Инвертируем маску телефона
    main_background_no_phone = cv2.bitwise_and(main_background, main_background, mask=phone_frame_mask_inv)
    main_background_no_phone = extract_green_layers(main_background_no_phone, global_editor_config.lower_green, global_editor_config.upper_green)

    # Объединяем изображение с удаленным зеленым слоем и областью телефона
    main_background = cv2.add(main_background_no_phone, cv2.bitwise_and(main_background, main_background, mask=phone_screen_mask))

    # Check if zooming should start
    if start_zooming:
        global_editor_config.zoom_scale += global_editor_config.zoom_increment
        main_background = apply_zoom_to_center(main_background, (x, y, w, h), background_frame, phone_frame, frame_width, frame_height, global_editor_config.zoom_scale)

    if not start_zooming and consecutive_frame_count >= required_frames_for_one_second:
        detected_for_required_period = True
        start_zooming = True
        print("Background detected for more than one second.")

    return main_background


def replace_phone_screen_video(image, background_frame, required_frames_for_one_second, frame_width, frame_height):
    global consecutive_frame_count
    global detected_for_required_period
    global start_zooming
    global global_editor_config

    # Преобразуем изображение в нужный формат для сегментации
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    result = mediapipe_segmentor.process(frame_rgb)
    segmentation_mask = result.segmentation_mask

    overlay_alpha = (segmentation_mask * 255).astype(np.uint8)
    overlay_color = image

    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv_frame, global_editor_config.lower_blue, global_editor_config.upper_blue)
    blurred_mask = cv2.GaussianBlur(blue_mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    blue_mask_cleaned = cv2.morphologyEx(blurred_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(blue_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image[blue_mask != 0] = [0, 0, 0]

    if not contours:
        main_background = apply_background(overlay_alpha, overlay_color, background_frame,
                                                                    frame_width, frame_height)
        return main_background

    filtered_contours = [c for c in contours if cv2.contourArea(c) > 1700]
    if not filtered_contours:
        main_background = apply_background(overlay_alpha, overlay_color, background_frame,
                                                                     frame_width, frame_height)
        return main_background

    largest_contour = max(filtered_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    background_phone_frame_resized = cv2.resize(background_frame, (w, h))

    roi = image[y:y + h, x:x + w]

    phone_screen_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.drawContours(phone_screen_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    phone_screen_mask_roi = phone_screen_mask[y:y + h, x:x + w]
    background_with_mask = cv2.bitwise_and(background_phone_frame_resized, background_phone_frame_resized,
                                               mask=phone_screen_mask_roi)
    inverse_mask = cv2.bitwise_not(phone_screen_mask_roi)
    roi_with_edges = cv2.bitwise_and(roi, roi, mask=inverse_mask)

    image[y:y + h, x:x + w] = cv2.add(roi_with_edges, background_with_mask)

    consecutive_frame_count += 1

    main_background = apply_background(overlay_alpha, overlay_color, background_frame,
                                                                frame_width, frame_height)

    if start_zooming:
        global_editor_config.zoom_scale += global_editor_config.zoom_increment
        main_background = apply_zoom_to_center(main_background, (x, y, w, h), background_frame, frame_width,
                                                frame_height, global_editor_config.zoom_scale)

    if not start_zooming and consecutive_frame_count >= required_frames_for_one_second:
        detected_for_required_period = True
        start_zooming = True
        print("Background detected for more than one second.")

    return main_background

def create_temp_video(source_path, output_dir, fps):
    base_name = os.path.splitext(os.path.basename(source_path))[0]
    temp_path = os.path.join(output_dir, f"{base_name}_temp.mp4")

    # Create temp video if it doesn't exist
    if not os.path.exists(temp_path):
        video_clip = mp.VideoFileClip(source_path).set_fps(fps)
        video_clip.write_videofile(temp_path, codec='libx264')

    return temp_path


def apply_blur(frame, blur_radius=10):
    """Apply a blur effect to a single video frame using PIL."""
    img = Image.fromarray(frame)
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    return np.array(img)


def create_resized_video(fps, target_width, target_height, duration=None, target='main_background'):
    global global_editor_config
    if target == 'phone_background':
        temp_video_path = os.path.join(global_editor_config.phone_background_dir)
        source_video = global_editor_config.phone_background
    else:
        temp_video_path = os.path.join(global_editor_config.full_background_dir)
        source_video = global_editor_config.full_background
    source_video_without_extension = os.path.splitext(os.path.basename(source_video))[0]
    temp_video = os.path.join(temp_video_path, f'{source_video_without_extension}_temp.mp4')

    if not os.path.exists(temp_video_path):
        os.makedirs(temp_video_path)

    # Create temp video if it doesn't exist
    if not os.path.exists(temp_video):
        # Load the main video
        video_clip = mp.VideoFileClip(source_video).set_fps(fps)

        # Target aspect ratio
        target_aspect_ratio = target_width / target_height
        video_aspect_ratio = video_clip.w / video_clip.h

        # Resize and crop while centering the video
        if video_aspect_ratio > target_aspect_ratio:
            new_width = int(target_aspect_ratio * video_clip.h)
            video_clip = video_clip.crop(x_center=video_clip.w // 2, width=new_width)
        else:
            new_height = int(video_clip.w / target_aspect_ratio)
            video_clip = video_clip.crop(y_center=video_clip.h // 2, height=new_height)

        # Resize the video to match the target size exactly
        video_clip = video_clip.resize((target_width, target_height))

        # Create blurred background
        background_clip = video_clip.fl_image(lambda frame: apply_blur(frame, blur_radius=10))

        # If duration is specified, adjust video duration
        if duration:
            # If video duration is shorter, loop it to match the target duration
            if video_clip.duration < duration:
                loops_needed = int(duration // video_clip.duration) + 1
                video_clip = mp.concatenate_videoclips([video_clip] * loops_needed)
            video_clip = video_clip.subclip(0, duration)
        else:
            # If no duration is specified, keep original duration
            duration = video_clip.duration

        # Adjust background to match the main video duration
        if background_clip.duration < duration:
            loops_needed = int(duration // background_clip.duration) + 1
            background_clip = mp.concatenate_videoclips([background_clip] * loops_needed)
            background_clip = background_clip.subclip(0, duration)
        else:
            background_clip = background_clip.subclip(0, duration)

        # Combine background and video, setting the exact duration for final video
        final_video = mp.CompositeVideoClip(
            [background_clip, video_clip.set_position("center")],
            size=(target_width, target_height)
        ).set_duration(duration)

        # Save the temporary video file
        final_video.write_videofile(temp_video, codec='libx264', fps=fps)

    return temp_video


# Main chroma replace function
def chroma_replace(editor_config):
    global global_editor_config
    global_editor_config = editor_config
    video = cv2.VideoCapture(global_editor_config.original_video)

    frame_width, frame_height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps

    # Создание директорий, если их нет
    if not os.path.exists(global_editor_config.temp_dir):
        os.makedirs(global_editor_config.temp_dir)

    # Пути для фонов
    temp_full_back_path = create_resized_video(
        fps,
        target_width=frame_width,
        target_height=frame_height,
        duration=duration
    )
    temp_phone_back_path = create_resized_video(
        fps,
        target_width=frame_width,
        target_height=frame_height,
        target='phone_background'
    )
    background_phone_video = cv2.VideoCapture(temp_phone_back_path)
    background_video = cv2.VideoCapture(temp_full_back_path)

    # Путь для выходного видео
    output_video = cv2.VideoWriter(
        global_editor_config.output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    required_frames_for_one_second = fps
    folder_path = global_editor_config.output_composition_path

    # Инициализация итератора кадров на основе типа вывода RVM
    if global_editor_config.robust_output_type == 'png':
        frame_iterator = FIter(path=folder_path)
    elif global_editor_config.robust_output_type == 'video':
        frame_iterator = VIter(path=os.path.join(folder_path, 'com.mp4'))
    else:
        raise TypeError("This type is not supported")

    ret_bg, background_phone_frame = background_phone_video.read()
    if not ret_bg:
        background_phone_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_bg, background_phone_frame = background_phone_video.read()

    for frame in frame_iterator:
        ret_bg, background_frame = background_video.read()
        if not ret_bg:
            background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_bg, background_frame = background_video.read()

        if start_phone_video:
            ret_bg, background_phone_frame = background_phone_video.read()
            if not ret_bg:
                background_phone_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_bg, background_phone_frame = background_phone_video.read()

        if global_editor_config.robust_output_type == 'png':
            processed_frame = replace_phone_screen_png(
                frame, background_frame, background_phone_frame,
                required_frames_for_one_second, frame_width, frame_height
            )
        elif global_editor_config.robust_output_type == 'video':
            processed_frame = replace_phone_screen_video(
                frame, background_frame,
                required_frames_for_one_second, frame_width, frame_height
            )

        output_video.write(processed_frame)

    while background_phone_video.isOpened():
        ret_bg, background_phone_frame = background_phone_video.read()
        if not ret_bg:
            break
        main_background = cv2.resize(background_phone_frame, (frame_width, frame_height))
        output_video.write(main_background)

    # Освобождаем ресурсы
    output_video.release()
    background_phone_video.release()
    cv2.destroyAllWindows()