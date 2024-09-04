from PIL import Image
import os

def split_image_by_color(image_path, output_dir_part1, output_dir_part2, blue_colors, red_colors, offset=35):
    # Открытие изображения и конвертация в RGB
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    pixels = image.load()

    blue_pixel_position = None
    red_pixel_found = False

    # Поиск последнего пикселя синего и наличие красного цвета
    for y in range(height):
        for x in range(width):
            pixel = pixels[x, y]
            if isinstance(pixel, tuple):
                if pixel in blue_colors:
                    blue_pixel_position = y
                if pixel in red_colors:
                    red_pixel_found = True
            else:
                print(f"Найден пиксель не в формате RGB: {pixel} на позиции {(x, y)}")

    # Если синий пиксель найден
    if blue_pixel_position is not None:
        split_position = blue_pixel_position + offset
        if split_position < height:
            upper_part = image.crop((0, 0, width, split_position))
            lower_part = image.crop((0, split_position, width, height))

            os.makedirs(output_dir_part1, exist_ok=True)
            os.makedirs(output_dir_part2, exist_ok=True)

            base_name = os.path.basename(image_path)
            upper_part.save(os.path.join(output_dir_part1, base_name))
            lower_part.save(os.path.join(output_dir_part2, base_name))
    else:
        # Сохранение изображений, содержащих только красный цвет, в папку 2,
        # или оставление изображений без красного и синего цветов в папку 1
        base_name = os.path.basename(image_path)
        if red_pixel_found:
            os.makedirs(output_dir_part2, exist_ok=True)
            image.save(os.path.join(output_dir_part2, base_name))
        else:
            os.makedirs(output_dir_part1, exist_ok=True)
            image.save(os.path.join(output_dir_part1, base_name))

def process_images_in_directory(input_dir, output_dir_part1, output_dir_part2, blue_colors, red_colors, offset=35):
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            split_image_by_color(image_path, output_dir_part1, output_dir_part2, blue_colors, red_colors, offset)

# Пример использования
offset = 55
input_dir = "C:/Users/Игорь/Desktop/pet-prodject/parser_edu/inp/output/Теория PNG/Теория по папкам/1"
output_dir_part1 = "C:/Users/Игорь/Desktop/pet-prodject/parser_edu/inp/output/Теория PNG/Теория по папкам/1/1"
output_dir_part2 = "C:/Users/Игорь/Desktop/pet-prodject/parser_edu/inp/output/Теория PNG/Теория по папкам/1/2"

# Массив RGB кодов для синего цвета
blue_colors = [(231, 242, 248)]
# Массив RGB кодов для красного цвета
red_colors = [(252, 232, 233)]

process_images_in_directory(input_dir, output_dir_part1, output_dir_part2, blue_colors, red_colors, offset)

