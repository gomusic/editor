import cv2
import numpy as np

# Определяем диапазон зеленого цвета в HSV
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

def refine_silhouette(mask):
    # Применяем морфологическое закрытие для устранения черных вкраплений внутри силуэта
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Применяем операцию заливки, чтобы заполнить оставшиеся черные области внутри силуэта
    # Инвертируем маску, чтобы чёрные области стали белыми, и наоборот
    inverted_mask = cv2.bitwise_not(mask)
    # Находим контуры
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Заливаем все внутренние контуры белым цветом
    cv2.drawContours(inverted_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Возвращаем обратно инвертированную маску
    mask = cv2.bitwise_not(inverted_mask)

    return mask

def extract_green_layers(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Создаем маску зеленого цвета
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Создаем инвертированную маску силуэта
    silhouette_mask = cv2.bitwise_not(green_mask)

    # Улучшаем силуэт: заполняем черные области белым и корректируем контуры
    refined_silhouette = refine_silhouette(silhouette_mask)

    # Усиливаем насыщенность зеленого цвета внутри силуэта
    hsv[:, :, 1][green_mask != 0] = np.clip(hsv[:, :, 1][green_mask != 0] * 1.4, 0, 255)

    # Создаем зеленый слой, состоящий из зелени вне силуэта
    green_layer = cv2.bitwise_and(frame, frame, mask=green_mask)

    # Удаляем зеленый цвет из исходного кадра (удаляем его вне силуэта)
    frame_no_green = cv2.bitwise_and(frame, frame, mask=refined_silhouette)

    # Возвращаем два результата:
    # 1) кадр с усиленным зеленым внутри силуэта и без зеленого снаружи
    # 2) зеленый слой, где вся зелень за пределами силуэта сохранена
    return frame_no_green, green_layer

# Открываем исходное видео
input_video = cv2.VideoCapture('output_test.mp4')
fps = int(input_video.get(cv2.CAP_PROP_FPS))
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Создаем видеофайлы для сохранения результатов
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('result.mp4', fourcc, fps, (frame_width, frame_height))
green_layers_video = cv2.VideoWriter('green_layers.mp4', fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = input_video.read()
    if not ret:
        break

    # Применяем нашу функцию для обработки каждого фрейма
    processed_frame, green_layer = extract_green_layers(frame)

    output_video.write(processed_frame)
    green_layers_video.write(green_layer)

input_video.release()
output_video.release()
green_layers_video.release()
cv2.destroyAllWindows()
