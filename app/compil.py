import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing

# Параметры прямоугольников для удаления частей изображения
rects = [
    (0, 425, 0, 10524),       # Левый прямоугольник
    (7001, 7442, 0, 10524),   # Правый прямоугольник
    (421, 2363, 9663, 9923),  # Прямоугольник 1
    (5823, 7067, 319, 761)    # Прямоугольник 2
]

# Цвета для замены на белый
colors_to_replace = [
    (233, 232, 252),  # Красный цвет
    (248, 242, 231),
    (202, 219, 235),
    (224, 233, 240),
    (224, 211, 220),
    (224, 232, 240),
    (254, 254, 254),
    (224, 232, 239),
    (253, 254, 254),
    (224, 232, 241),
    (251, 232, 233),
    (251, 252, 253),
    (224, 211, 219),
    (255, 254, 254),
    (253, 253, 254),
    (255, 254, 255),
    (251, 231, 232),
    (254, 254, 255),
    (227, 235, 241),
    (225, 233, 240),
    (225, 234, 240),
    (224, 232, 238),
    (224, 210, 219),
    (252, 253, 254),
    (252, 253, 253),
    (226, 234, 241),
    (247, 249, 251),
    (225, 211, 218),
    (254, 255, 255),
    (248, 250, 252),
    (233, 239, 244),
    (250, 251, 252),
    (248, 250, 251),
    (249, 251, 252),
    (246, 248, 250),
    (255, 253, 253),
    (231, 238, 243),
    (252, 252, 253),
    (249, 250, 252),
    (224, 210, 218),
    (230, 237, 242),
    (244, 247, 249),
    (229, 236, 242),
    (255, 253, 254),
    (241, 245, 248),
    (226, 235, 241),
    (223, 232, 241),
    (245, 248, 250),
    (243, 246, 249),
    (240, 244, 247),
    (226, 234, 240),
    (250, 231, 233),
    (228, 236, 242),
    (240, 244, 248),
    (244, 247, 250),
    (232, 238, 243),
    (237, 242, 246),
    (228, 235, 241),
    (239, 243, 247),
    (248, 249, 251),
    (250, 231, 232),
    (245, 247, 250),
    (254, 249, 249),
    (224, 211, 218),
    (250, 251, 253),
    (236, 241, 245),
    (202, 219, 235),
    (201, 218, 234),
    (200, 218, 234),
    ]


def replace_pixels_with_white(image_path, output_path, rects, colors_to_replace):
    # Загружаем изображение с использованием OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return

    # Преобразуем цвета в numpy массив
    for color in colors_to_replace:
        lower_bound = np.array(color) - 10  # Диапазон цветовых значений
        upper_bound = np.array(color) + 10
        mask = cv2.inRange(img, lower_bound, upper_bound)
        img[mask > 0] = (255, 255, 255)

    # Обработка прямоугольников
    for rect in rects:
        x_start, x_end, y_start, y_end = rect
        img[y_start:y_end, x_start:x_end] = (255, 255, 255)

    # Сохраняем результат в новую папку
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)

def classify_and_process_image(image_path, output_base_dir):
    # Загружаем изображение
    img = cv2.imread(image_path)
    
    if img is None or img.size == 0:
        print(f"Ошибка загрузки изображения: {image_path}")
        return f"Ошибка загрузки изображения: {image_path}"

    # Определяем нужные цвета
    cover_color = np.array([189, 121, 20])  # Цвет обложки
    red_color = np.array([233, 232, 252])  # Красный цвет
    blue_color = np.array([248, 242, 231])  # Синий цвет
    
    # Создаем маски для каждого из нужных цветов
    mask_red = cv2.inRange(img, red_color, red_color)
    mask_blue = cv2.inRange(img, blue_color, blue_color)
    mask_cover = cv2.inRange(img, cover_color, cover_color)
    
    red_pixels = cv2.countNonZero(mask_red)
    blue_pixels = cv2.countNonZero(mask_blue)
    cover_pixels = cv2.countNonZero(mask_cover)
    
    base_name = os.path.basename(image_path)
    
    if red_pixels > 0 and blue_pixels > 0:
        save_image_parts(image_path, output_base_dir, 'pptx', 'word', 4113, 4113)
        return f"{base_name}: Красный и синий цвета"
    elif cover_pixels > 0:
        save_image_parts(image_path, output_base_dir, 'pptx', 'word', 3913, 0, is_cover=True)
        return f"{base_name}: Обложка"
    elif red_pixels > 0:
        save_image_parts(image_path, output_base_dir, None, 'word')
        return f"{base_name}: Красный цвет"
    elif blue_pixels > 0:
        save_image_parts(image_path, output_base_dir, 'pptx', None, 4113)
        return f"{base_name}: Синий цвет"
    else:
        return f"{base_name}: не смог определить"

def save_image_parts(image_path, output_base_dir, pptx_folder, word_folder, pptx_height=None, word_height=None, is_cover=False):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    pptx_output_dir = os.path.join(output_base_dir, pptx_folder) if pptx_folder else None
    word_output_dir = os.path.join(output_base_dir, word_folder) if word_folder else None

    if is_cover:
        upper_part = image[0:pptx_height, 0:width]
        save_image(pptx_output_dir, upper_part, image_path)
        save_image(word_output_dir, image, image_path)
    elif pptx_height and word_height:
        upper_part = image[0:pptx_height, 0:width]
        lower_part = image[pptx_height:height, 0:width]
        save_image(pptx_output_dir, upper_part, image_path)
        save_image(word_output_dir, lower_part, image_path)
    elif pptx_height:
        upper_part = image[0:pptx_height, 0:width]
        save_image(pptx_output_dir, upper_part, image_path)
    elif word_height:
        lower_part = image[word_height:height, 0:width]
        save_image(word_output_dir, lower_part, image_path)
    else:
        save_image(word_output_dir, image, image_path)

def save_image(output_dir, image_part, image_path):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(image_path)
        cv2.imwrite(os.path.join(output_dir, base_name), image_part)

def get_all_image_paths(root_folder):
    image_paths = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def classify_images_in_folder(folder_path, output_base_dir):
    image_paths = get_all_image_paths(folder_path)
    results = []

    max_workers = max(1, int(multiprocessing.cpu_count() * 0.75))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for image_path in image_paths:
            subfolder = os.path.basename(os.path.dirname(image_path))
            futures[executor.submit(classify_and_process_image, image_path, os.path.join(output_base_dir, subfolder))] = image_path

        for future in tqdm(as_completed(futures), total=len(image_paths)):
            image_path = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Ошибка при обработке {image_path}: {e}")
    
    return results

def replace_colors_in_output_folder(output_base_dir, colors_to_replace):
    image_paths = get_all_image_paths(output_base_dir)
    output_dir = os.path.join(output_base_dir, 'processed')

    max_workers = max(1, int(multiprocessing.cpu_count() * 0.75))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for img in image_paths:
            output_img = os.path.join(output_dir, os.path.relpath(img, output_base_dir))
            futures[executor.submit(replace_pixels_with_white, img, output_img, rects, colors_to_replace)] = img

        for future in tqdm(as_completed(futures), total=len(futures), desc="Replacing colors"):
            try:
                future.result()
            except Exception as exc:
                img = futures[future]
                print(f"Ошибка при обработке файла {img}: {exc}")

if __name__ == '__main__':
    folder_path = r'D:\edu\Teor_kach'
    output_base_dir = r'D:\edu\Teor_output_folder'

    results = classify_images_in_folder(folder_path, output_base_dir)

    # После завершения обработки, выполняем замену цветов в папке output
    replace_colors_in_output_folder(output_base_dir, colors_to_replace)
