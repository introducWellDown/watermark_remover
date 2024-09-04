import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from numba import jit, prange
import CONST

# Цвета для замены на белый
colors_to_replace = CONST.colors_to_replace

def get_all_image_paths(root_folder):
    """Получение всех путей к изображениям в папке, включая вложенные папки."""
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
    """Замена указанных цветов на белый с использованием Numba."""
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

def replace_colors_in_images(folder_path, output_base_dir, colors_to_replace):
    """Проходит по всем изображениям и заменяет указанные цвета на белый с сохранением структуры папок."""
    image_paths = get_all_image_paths(folder_path)
    
    max_workers = max(1, int(multiprocessing.cpu_count() * 0.20))  # Оптимальное количество процессов
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for img in image_paths:
            relative_path = os.path.relpath(img, folder_path)  # Получаем относительный путь
            output_img = os.path.join(output_base_dir, relative_path)  # Сохраняем структуру
            futures[executor.submit(replace_pixels_with_white, img, output_img, colors_to_replace)] = img
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Replacing colors in images"):
            try:
                future.result()
            except Exception as exc:
                img = futures[future]
                print(f"Ошибка при обработке файла {img}: {exc}")

if __name__ == '__main__':
    folder_path = r'D:\edu\final_teor'
    output_base_dir = r'D:\edu\true_miror'

    # Заменяем указанные цвета на белый в изображениях с использованием ускорений и сохранением структуры папок
    replace_colors_in_images(folder_path, output_base_dir, colors_to_replace)
