from PIL import Image

def count_pixels(image_path):
    image = Image.open(image_path)
    pixels = image.getdata()
    pixel_counts = {}

    for pixel in pixels:
        # Игнорирование значения прозрачности (если оно есть)
        pixel = pixel[:3]
        if pixel in pixel_counts:
            pixel_counts[pixel] += 1
        else:
            pixel_counts[pixel] = 1

    sorted_counts = sorted(pixel_counts.items(), key=lambda x: x[1], reverse=True)

    for color, count in sorted_counts:
        if count > 200:
            # Форматированный вывод без значения прозрачности и количества пикселей
            print(f"{color}, {count}")

# Пример использования
count_pixels("C:/Users/Игорь/Desktop/pet-prodject/parser_edu/inp/input/Снимок10.PNG")
