import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import CONST


def replace_pixels_with_white(image_path, output_path, colors_to_replace, tolerance=10):
    """Замена указанных цветов на белый с учетом полупрозрачности."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return

    if img.shape[2] == 4:  # Проверка наличия альфа-канала
        alpha_channel = img[:, :, 3]
        img_rgb = img[:, :, :3]
    else:
        alpha_channel = None
        img_rgb = img

    # Преобразуем список цветов в массив NumPy
    colors_to_replace = np.array(colors_to_replace)

    # Создаем маску для каждого цвета в colors_to_replace с учетом полупрозрачности
    mask = np.zeros(img_rgb.shape[:2], dtype=bool)
    for color in colors_to_replace:
        # Вычисляем евклидово расстояние между пикселями и целевым цветом
        dist = np.linalg.norm(img_rgb - color, axis=-1)
        mask |= dist <= tolerance

    # Применяем маску, заменяя соответствующие пиксели на белый цвет
    img_rgb[mask] = [255, 255, 255]

    if alpha_channel is not None:
        img_rgba = np.dstack((img_rgb, alpha_channel))
        output_img = img_rgba
    else:
        output_img = img_rgb

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output_img)

def process_images_with_multiprocessing(input_dir, output_dir, colors_to_replace, tolerance=10):
    """
    Функция для замены цветов в изображениях с использованием мультипроцессинга.
    """
    image_paths = get_all_image_paths(input_dir)
    
    max_workers = max(1, int(multiprocessing.cpu_count() * 0.9))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for img_path in image_paths:
            output_img_path = os.path.join(output_dir, os.path.relpath(img_path, input_dir))
            futures[executor.submit(replace_pixels_with_white, img_path, output_img_path, colors_to_replace, tolerance)] = img_path

        for future in tqdm(as_completed(futures), total=len(futures), desc="Replacing colors"):
            try:
                future.result()
            except Exception as exc:
                img_path = futures[future]
                print(f"Ошибка при обработке файла {img_path}: {exc}")

def get_all_image_paths(root_folder):
    """
    Получение списка путей ко всем изображениям в директории.
    """
    image_paths = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def fill_rectangles(image_path, output_path, rects):
    """Заливаем прямоугольные области изображения белым цветом."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return

    for rect in rects:
        x_start, x_end, y_start, y_end = rect
        img[y_start:y_end, x_start:x_end] = (255, 255, 255)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)

def classify_and_process_image(image_path, output_base_dir):
    """Классификация изображения на основе цвета и разрезка на части."""
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
    """Сохранение разрезанных частей изображения в соответствующие папки."""
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
    """Сохранение изображения в заданную папку."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(image_path)
        cv2.imwrite(os.path.join(output_dir, base_name), image_part)

def process_folder_with_rectangles(folder_path, output_base_dir, rects):
    """Заливка прямоугольников и последующая классификация изображений."""
    image_paths = get_all_image_paths(folder_path)
    results = []

    max_workers = max(1, int(multiprocessing.cpu_count() * 0.9))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        fill_futures = {}
        for image_path in image_paths:
            subfolder = os.path.basename(os.path.dirname(image_path))
            output_folder = os.path.join(output_base_dir, subfolder)
            filled_image_path = os.path.join(output_folder, 'filled_' + os.path.basename(image_path))

            # Сначала выполняем заливку прямоугольников белым
            fill_futures[executor.submit(fill_rectangles, image_path, filled_image_path, rects)] = filled_image_path

        # Ожидаем завершения заливки
        for future in tqdm(as_completed(fill_futures), total=len(image_paths), desc="Filling rectangles"):
            try:
                future.result()
            except Exception as exc:
                image_path = fill_futures[future]
                print(f"Ошибка при заливке прямоугольников в файле {image_path}: {exc}")
        
        # После завершения заливки выполняем классификацию и разрезку
        classify_futures = {}
        for filled_image_path in fill_futures.values():
            subfolder = os.path.basename(os.path.dirname(filled_image_path))
            classify_futures[executor.submit(classify_and_process_image, filled_image_path, os.path.join(output_base_dir, subfolder))] = filled_image_path

        for future in tqdm(as_completed(classify_futures), total=len(classify_futures), desc="Classifying and processing"):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                image_path = classify_futures[future]
                print(f"Ошибка при классификации и обработке файла {image_path}: {exc}")

    return results

def replace_colors_in_processed_folders(output_base_dir, colors_to_replace, tolerance=15):
    """Проходимся по папкам pptx и word и заменяем указанные цвета на белый."""
    for root, dirs, files in os.walk(output_base_dir):
        for folder in dirs:
            if folder in ['pptx', 'word']:
                folder_path = os.path.join(root, folder)
                process_images_with_multiprocessing(folder_path, folder_path, colors_to_replace, tolerance)

if __name__ == '__main__':
    folder_path = r'D:\edu\Teor_kach'
    output_base_dir = r'D:\edu\Teor_output_folder'

    # Прямоугольники для заливки
    rects = [
        (0, 425, 0, 10524),       # Левый прямоугольник
        (7001, 7442, 0, 10524),   # Правый прямоугольник
        (421, 2363, 9663, 9923),  # Прямоугольник 1
        (5823, 7067, 319, 761)    # Прямоугольник 2
    ]

    # Цвета для замены на белый
    colors_to_replace = CONST.colors_to_replace
    # Этап 1: Заливаем прямоугольники белым и затем классифицируем изображения
    results = process_folder_with_rectangles(folder_path, output_base_dir, rects)

    # Этап 2: Проходимся по папкам pptx и word и заменяем указанные цвета на белый
    replace_colors_in_processed_folders(output_base_dir, colors_to_replace, tolerance=15)
