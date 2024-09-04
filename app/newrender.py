import fitz  # PyMuPDF
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def convert_page_to_png(pdf_path, page_num, output_dir, dpi=200):
    # Открытие PDF документа и конвертация страницы в PNG
    pdf_document = fitz.open(pdf_path)
    zoom = dpi / 72  # Увеличение для получения нужного DPI (72 DPI - стандартное разрешение)
    mat = fitz.Matrix(zoom, zoom)
    
    page = pdf_document.load_page(page_num)
    pix = page.get_pixmap(matrix=mat)
    
    output_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
    pix.save(output_path)
    return page_num + 1

def convert_pdf_to_png(pdf_path, output_dir, dpi=200):
    # Проверка и создание выходной директории
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Открытие PDF документа
    pdf_document = fitz.open(pdf_path)
    num_pages = len(pdf_document)

    # Параллельная обработка страниц
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(convert_page_to_png, pdf_path, page_num, output_dir, dpi) for page_num in range(num_pages)]
        
        for future in tqdm(as_completed(futures), total=num_pages, desc="Converting pages"):
            page_num = future.result()


if __name__ == '__main__':
    # Пример использования
    pdf_path = "D:/edu/AL-1701-5 Практика с ответами.pdf"
    output_dir = "D:/edu/prak_otv"  # Замените на путь к выходной директории
    dpi = 900  # Разрешение в DPI (можно увеличить для лучшего качества)

    convert_pdf_to_png(pdf_path, output_dir, dpi)
