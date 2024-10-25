import cv2
import mediapipe as mp
import numpy as np
from iterators.video_iterator import VideoIterator as VIter


iterator = VIter('elements.mp4')


def zoom():
    # Загружаем шаблон
    template = cv2.imread('./src/share/big-share.png')

    # Получаем ширину и высоту первого кадра для настроек видео
    first_frame = next(iterator)
    height, width, _ = first_frame.shape

    # Настраиваем видеозапись (имя файла, кодек, FPS, размер кадра)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для .mp4
    output_video = cv2.VideoWriter('output_video_contours_share_zoom_test.mp4', fourcc, 20.0, (width, height))

    # Инициализация переменных
    previous_best_match = None
    previous_best_val = 0
    previous_best_scale = 1.0
    search_window_size = 50
    count = 0

    # Настройки затемнения
    max_darkness = 0.8

    # Настройки зуммирования
    zoom_start_frame = 15
    zoom_duration = 30
    max_zoom_factor = 5.0

    # Добавляем флаг для обратного зуммирования и флаг окончания зума
    zoom_reverse_duration = 20  # Количество кадров для обратного зуммирования

    for frame in iterator:
        print(count)
        if True:
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            _, template_thresh = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            w, h = template_gray.shape[::-1]

            # Поиск шаблона
            if previous_best_val > 0.7:
                best_match = None
                best_val = 0

                previous_x, previous_y = previous_best_match[0]

                x_min = max(0, previous_x - search_window_size)
                x_max = min(image_gray.shape[1] - w, previous_x + search_window_size)
                y_min = max(0, previous_y - search_window_size)
                y_max = min(image_gray.shape[0] - h, previous_y + search_window_size)

                roi = image_gray[y_min:y_max, x_min:x_max]

                resized_template = cv2.resize(template_gray,
                                              (int(w * previous_best_scale), int(h * previous_best_scale)))

                if resized_template.shape[0] <= roi.shape[0] and resized_template.shape[1] <= roi.shape[1]:
                    result = cv2.matchTemplate(roi, resized_template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                    if max_val > best_val:
                        best_val = max_val
                        best_match = (max_loc, previous_best_scale)

                if best_val < 0.7:
                    previous_best_val = 0
                    continue

                previous_best_match = (best_match[0][0] + x_min, best_match[0][1] + y_min)
                previous_best_val = best_val

            else:
                best_match = None
                best_val = 0

                for scale in np.linspace(0.8, 1.2, 20):
                    resized_template = cv2.resize(template_gray, (int(w * scale), int(h * scale)))

                    if resized_template.shape[0] > image_gray.shape[0] or resized_template.shape[1] > image_gray.shape[
                        1]:
                        continue

                    result = cv2.matchTemplate(image_gray, resized_template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                    if max_val > best_val:
                        best_val = max_val
                        best_match = (max_loc, scale)

                previous_best_match = best_match
                previous_best_val = best_val
                previous_best_scale = best_match[1]

            threshold = 0.8

            if best_match and best_val >= threshold:
                max_loc, scale = best_match
                top_left = max_loc
                w_scaled, h_scaled = int(w * scale), int(h * scale)

                mask = np.ones_like(frame, dtype=np.uint8) * 255



                # Плавное зуммирование (вход в зум)
                if zoom_start_frame <= count < (zoom_start_frame + zoom_duration):

                    # Плавное затемнение
                    alpha = (count / zoom_duration) * max_darkness
                    alpha = min(alpha, max_darkness)

                    if alpha > 0:
                        darkened_frame = cv2.addWeighted(frame,1 - alpha, np.zeros_like(frame), alpha, 0)

                        for contour in contours:
                            scaled_contour = contour * scale
                            scaled_contour += np.array(top_left)

                            cv2.drawContours(mask, [scaled_contour.astype(int)], -1, (0, 0, 0), thickness=cv2.FILLED)

                        frame = np.where(mask == 0, frame, darkened_frame)

                    zoom_factor = 1.0 + (max_zoom_factor - 1.0) * (((count - zoom_start_frame) / zoom_duration) ** 2)

                    center_x = top_left[0] + w_scaled // 2
                    center_y = top_left[1] + h_scaled // 2

                    new_w = int(width / zoom_factor)
                    new_h = int(height / zoom_factor)

                    new_top_left_x = max(0, center_x - new_w // 2)
                    new_top_left_y = max(0, center_y - new_h // 2)

                    new_bottom_right_x = min(new_top_left_x + new_w, width)
                    new_bottom_right_y = min(new_top_left_y + new_h, height)

                    roi_zoomed = frame[new_top_left_y:new_bottom_right_y, new_top_left_x:new_bottom_right_x]
                    frame = cv2.resize(roi_zoomed, (width, height))

                # Когда зум достиг максимума
                elif count == zoom_start_frame + zoom_duration:
                    last_frame = frame.copy()  # Сохраняем последний кадр
                    reverse_count = 0  # Счётчик обратного зума

                    # Обратное зуммирование
                    while reverse_count < zoom_reverse_duration:
                        reverse_zoom_factor = max_zoom_factor - (max_zoom_factor - 1.0) * (
                                    reverse_count / zoom_reverse_duration)
                        zoom_factor = reverse_zoom_factor

                        center_x = top_left[0] + w_scaled // 2
                        center_y = top_left[1] + h_scaled // 2

                        new_w = int(width / zoom_factor)
                        new_h = int(height / zoom_factor)

                        new_top_left_x = max(0, center_x - new_w // 2)
                        new_top_left_y = max(0, center_y - new_h // 2)

                        new_bottom_right_x = min(new_top_left_x + new_w, width)
                        new_bottom_right_y = min(new_top_left_y + new_h, height)

                        roi_zoomed = last_frame[new_top_left_y:new_bottom_right_y, new_top_left_x:new_bottom_right_x]
                        frame = cv2.resize(roi_zoomed, (width, height))

                        if reverse_count < zoom_reverse_duration:
                            reverse_alpha = max_darkness - (reverse_count / zoom_reverse_duration) * max_darkness
                        else:
                            reverse_alpha = 0

                        if reverse_alpha < max_darkness:
                            brightened_frame = cv2.addWeighted(frame, 1 - reverse_alpha, np.zeros_like(frame),
                                                               reverse_alpha, 0)

                            # Объединение осветленного фрейма с исходным на основе маски
                            frame = np.where(mask == 0, frame, brightened_frame)

                        reverse_count += 1
                        output_video.write(frame)

                output_video.write(frame)
                print('Еще работаю. Фрейм: ', count)
        count += 1

    output_video.release()
    print("Video saved as 'output_video_contours_share_zoom_test.mp4'.")


if __name__ == ('__main__'):
    zoom()