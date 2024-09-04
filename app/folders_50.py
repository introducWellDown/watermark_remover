import os

def create_folders(base_dir, num_folders=8):
    # Создание 50 папок с именами от 1 до 50
    for i in range(1, num_folders + 1):
        folder_name = os.path.join(base_dir, str(i))
        os.makedirs(folder_name, exist_ok=True)
        print(f'Создана папка: {folder_name}')

# Пример использования
base_dir = r'D:\edu\prak_otv'  # Замените на путь к вашей директории
create_folders(base_dir)
