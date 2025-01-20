import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from tqdm import tqdm

def ensure_directories_exist(path_to_save):
    """Создает необходимые директории, если они не существуют."""
    for subdir in ['Clear', 'Mask', 'Text']:
        os.makedirs(os.path.join(path_to_save, subdir), exist_ok=True)

def get_random_font():
    font_type = ['NotoSansJP-Bold', 'OtsutomeFont_Ver3_16']
    return random.choice(font_type)

def crop_random_square(images, width, height):
    img = random.choice(images)
    width_img, height_img = img.size
    if width > width_img or height > height_img:
        raise ValueError("Размер квадрата больше, чем размеры изображения.")
    x = random.randint(0, width_img - width)
    y = random.randint(0, height_img - height)
    cropped_img = img.crop((x, y, x + width, y + height))
    return cropped_img

def generate_random_japanese_text(length):
    hiragana_range = (0x3040, 0x309F)
    katakana_range = (0x30A0, 0x30FF)
    kanji_range = (0x4E00, 0x9FAF)
    random_text = []
    for _ in range(length):
        char_type = random.choice(['hiragana', 'katakana', 'kanji'])
        if char_type == 'hiragana':
            char = chr(random.randint(*hiragana_range))
        elif char_type == 'katakana':
            char = chr(random.randint(*katakana_range))
        else:
            char = chr(random.randint(*kanji_range))
        random_text.append(char)
    return ''.join(random_text)

def generate_noise_image(width, height):
    noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    return Image.fromarray(noise, mode='L')

def generate_black_image(width, height):
    black_image = np.zeros((height, width), dtype=np.uint8)
    return Image.fromarray(black_image, mode='L')



def add_rectangle_to_image(image, position, size, color):
    draw = ImageDraw.Draw(image)
    draw.rectangle([position, (position[0] + size[0], position[1] + size[1])], fill=color)

def add_text_to_image(image, text, font_path, font_size, position, text_color, outline_color=(255, 255, 255), outline_width=3):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)

    # Рисуем обводку
    x, y = position
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:  # Не рисуем в центре
                draw.text((x + dx, y + dy), text, fill=outline_color, font=font)

    # Рисуем основной текст
    draw.text(position, text, fill=text_color, font=font)

def generate_images(n, width, height, text_length, font_path, font_size_range, path_to_save, mangas):
    ensure_directories_exist(path_to_save)
    for i in tqdm(range(n)):
        font_size = random.randint(font_size_range[0], font_size_range[1])
        noise_image = crop_random_square(mangas, width, height)
        black_image = generate_black_image(width, height)
        text = generate_random_japanese_text(random.randint(0, text_length))
        text2 = generate_random_japanese_text(random.randint(0, text_length))

        position = (random.randint(-font_size * 3, width - font_size * 2), random.randint(0 - font_size * 2, height))
        second_line_position = (random.randint(-font_size * 3, width - font_size * 2), random.randint(0 - font_size * 2, height))

        text_color_choice = random.choice(['dark', 'white'])
        if text_color_choice == 'dark':
            text_color = random.randint(0, 30)  # Тёмный текст
            outline_color = (255, 255, 255)  # Белая обводка
        else:
            text_color = random.randint(230, 255) # Белый текст
            outline_color = (0, 0, 0)  # Белая обводка

        text_size = (font_size * text_length, font_size + 10)
        text_size2 = (font_size * text_length, font_size + 10)

        noise_image.save(f'{path_to_save}/Clear/image_{i + 1}.png')

        add_rectangle_to_image(black_image, position, text_size, 255)
        add_rectangle_to_image(black_image, second_line_position, text_size2, 255)

        add_text_to_image(
            noise_image, text, font_path, font_size, position,
            (text_color,text_color,text_color), outline_color, outline_width=random.randint(0, 4)
        )

        add_text_to_image(
            noise_image, text2, font_path, font_size, second_line_position,
            (text_color,text_color,text_color), outline_color, outline_width=random.randint(0, 4)
        )

        black_image.save(f'{path_to_save}/Mask/image_{i + 1}.png')
        noise_image.save(f'{path_to_save}/Text/image_{i + 1}.png')


def main():
    parser = argparse.ArgumentParser(description="Генерация изображений с текстом.")
    parser.add_argument("--count", type=int, default=10000, help="Количество изображений (по умолчанию: 10000).")
    parser.add_argument("--width", type=int, default=128, help="Ширина изображений (по умолчанию: 128).")
    parser.add_argument("--height", type=int, default=128, help="Высота изображений (по умолчанию: 128).")
    parser.add_argument("--text_length", type=int, default=5, help="Длина текста (по умолчанию: 5).")
    parser.add_argument("--font_path", type=str, default="./NotoSansJP-Bold.ttf", help="Путь к файлу шрифта (по умолчанию: ./NotoSansJP-Bold.ttf).")
    parser.add_argument("--font_size_min", type=int, default=10, help="Минимальный размер шрифта (по умолчанию: 10).")
    parser.add_argument("--font_size_max", type=int, default=40, help="Максимальный размер шрифта (по умолчанию: 40).")
    parser.add_argument("--path_to_save", type=str, default="./train_data/", help="Путь для сохранения изображений (по умолчанию: ./DeepEraser/train_data/).")
    parser.add_argument("--manga_path", type=str, default="./clm", help="Путь к изображениям манги (по умолчанию: ./clm).")

    args = parser.parse_args()

    filenames = os.listdir(args.manga_path)
    images = [Image.open(os.path.join(args.manga_path, file)) for file in filenames]

    generate_images(
        args.count,
        args.width,
        args.height,
        args.text_length,
        args.font_path,
        (args.font_size_min, args.font_size_max),
        args.path_to_save,
        images
    )

if __name__ == "__main__":
    main()
