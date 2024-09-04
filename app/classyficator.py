import cv2
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def classify_image(image_path):
    # Проверка существования файла
    if not os.path.exists(image_path):
        return f"Файл не найден: {image_path}"

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
    
    # Классификация на основе количества пикселей соответствующих цветов
    if red_pixels > 0 and blue_pixels > 0:
        return "Красный и синий цвета"
    elif cover_pixels > 28054:
        return "Обложка"
    elif red_pixels > 0:
        return "Красный цвет"
    elif blue_pixels > 0:
        return "Синий цвет"
    else:
        return "не смог определить"

def get_all_image_paths(root_folder):
    image_paths = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def classify_images_in_folder(folder_path):
    image_paths = get_all_image_paths(folder_path)
    results = []

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(classify_image, image_path): image_path for image_path in image_paths}
        for future in tqdm(as_completed(futures), total=len(image_paths)):
            image_path = futures[future]
            try:
                result = future.result()
                results.append((os.path.basename(image_path), result))
            except Exception as e:
                print(f"Ошибка при обработке {image_path}: {e}")
    
    return results

if __name__ == '__main__':
    folder_path = r'D:\edu\Teor_nice_clear_folder'
    results = classify_images_in_folder(folder_path)

    # Сортируем результаты по имени файла
    results.sort(key=lambda x: int(x[0].split('_')[1].split('.')[0]))

    for filename, classification in results:
        print(f"{filename}: {classification}")
