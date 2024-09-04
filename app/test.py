import os
import cv2
import numpy as np
from tqdm import tqdm
import CONST
from numba import jit, prange

colors_to_replace = CONST.colors_to_replace

def calculate_rects_first_page(image_width, image_height):
    """Вычисляет параметры прямоугольников для удаления всей нижней линии и двух вертикальных линий на первой странице."""
    # Нижняя часть по всей ширине изображения
    line_height = int(image_height // 11.2)  # Высота нижней линии, которую нужно удалить
    y_bottom = image_height - line_height  # Начальная точка по Y для удаления нижней линии
    
    # Прямоугольник для нижней линии (вся ширина)
    rect_bottom = (0, image_width, y_bottom, image_height)
    
    # Левая и правая вертикальные линии
    x_left = int(image_width // 16.875)
    x_right = int(image_width - image_width // 16.875)
    rect_left = (0, x_left, 0, image_height)  # Левая вертикальная линия
    rect_right = (x_right, image_width, 0, image_height)  # Правая вертикальная линия
    
    return [rect_bottom, rect_left, rect_right]

def calculate_rects_other_pages(image_width, image_height):
    """Вычисляет параметры прямоугольников для остальных страниц."""
    # Этап 1: Заливаем белым полоски по краям
    x_left = int(image_width // 16.875)
    x_right = int(image_width - image_width // 16.875)
    rect1 = (0, x_left, 0, image_height)  # Левая часть
    rect2 = (x_right, image_width, 0, image_height)  # Правая часть
    
    # Этап 2: Заливаем левую нижнюю часть
    x_mid = image_width // 2
    y_bottom = int(image_height - image_height // 11.5)
    rect3 = (0, x_mid, y_bottom, image_height)
    
    return [rect1, rect2, rect3]

def fill_rectangles(image_path, output_path, rects):
    """Заполнение прямоугольников белым цветом."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return

    for rect in rects:
        x_start, x_end, y_start, y_end = rect
        img[y_start:y_end, x_start:x_end] = (255, 255, 255)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)

def get_all_image_paths(root_folder):
    """Получение всех путей к изображениям в папке."""
    image_paths = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

@jit(nopython=True, parallel=True)
def replace_pixels_with_white_numba(img_rgb, colors_to_replace, tolerance):
    """Замена указанных цветов на белый с использованием Numba для ускорения."""
    height, width, _ = img_rgb.shape
    for i in prange(height):
        for j in prange(width):
            for color in colors_to_replace:
                dist = np.sqrt(np.sum((img_rgb[i, j] - color) ** 2))
                if dist <= tolerance:
                    img_rgb[i, j, 0] = 255
                    img_rgb[i, j, 1] = 255
                    img_rgb[i, j, 2] = 255
                    break

    return img_rgb

def replace_pixels_with_white(image_path, output_path, colors_to_replace, tolerance=30):
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

    colors_to_replace = np.array(colors_to_replace, dtype=np.int32)
    
    # Используем Numba для ускоренной замены цветов
    img_rgb = replace_pixels_with_white_numba(img_rgb, colors_to_replace, tolerance)

    if alpha_channel is not None:
        img_rgba = np.dstack((img_rgb, alpha_channel))
        output_img = img_rgba
    else:
        output_img = img_rgb

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output_img)

def process_folder_with_rectangles(folder_path, output_base_dir):
    """Заливка прямоугольников белым с динамическими параметрами."""
    image_paths = get_all_image_paths(folder_path)
    results = []

    for idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
        subfolder = os.path.basename(os.path.dirname(image_path))
        output_folder = os.path.join(output_base_dir, subfolder)
        filled_image_path = os.path.join(output_folder, 'filled_' + os.path.basename(image_path))

        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Ошибка загрузки изображения: {image_path}")
                continue
            image_height, image_width = img.shape[:2]

            # Определяем, какие прямоугольники использовать
            if idx == 0:
                rects = calculate_rects_first_page(image_width, image_height)
            else:
                rects = calculate_rects_other_pages(image_width, image_height)
            
            fill_rectangles(image_path, filled_image_path, rects)
        except Exception as exc:
            print(f"Ошибка при заливке прямоугольников в файле {image_path}: {exc}")
            continue

    return results

def replace_colors_in_processed_folders(output_base_dir, colors_to_replace):
    """Замена указанных цветов на белый в папках pptx и word."""
    for root, dirs, files in os.walk(output_base_dir):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            image_paths = get_all_image_paths(folder_path)
            
            for img in tqdm(image_paths, desc=f"Replacing colors in {folder}"):
                output_img = os.path.join(folder_path, os.path.basename(img))
                try:
                    replace_pixels_with_white(img, output_img, colors_to_replace)
                except Exception as exc:
                    print(f"Ошибка при обработке файла {img}: {exc}")

if __name__ == '__main__':
    folder_path = r'D:\edu\prak_otv'
    output_base_dir = r'D:\edu\prak_otv_kach'

    # Этап 1: Заливаем прямоугольники белым с динамическими параметрами
    results = process_folder_with_rectangles(folder_path, output_base_dir)

    # Этап 2: Проходимся по папкам pptx и word и заменяем указанные цвета на белый
    replace_colors_in_processed_folders(output_base_dir, colors_to_replace)
