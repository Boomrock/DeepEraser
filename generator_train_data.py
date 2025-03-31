import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
from tqdm import tqdm


def ensure_directories(path_to_save):
    for subdir in ['Clear', 'Mask', 'Text']:
        os.makedirs(os.path.join(path_to_save, subdir), exist_ok=True)


def crop_random_region(images, width, height):
    img = random.choice(images)
    x = random.randint(0, img.width - width)
    y = random.randint(0, img.height - height)
    return img.crop((x, y, x + width, y + height))


def generate_japanese_text(max_length):
    def get_char():
        ranges = [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FAF)]
        return chr(random.randint(*random.choice(ranges)))

    return ''.join(get_char() for _ in range(random.randint(1, max_length)))


def generate_noise_mask(width, height, alpha=0.3):
    """Генерирует RGBA-маску шума с прозрачностью"""
    # Генерируем шум во всех четырех каналах
    noise = np.zeros((height, width, 4), dtype=np.uint8)

    # RGB каналы: нормальное распределение со средним 0 и СКО 255
    noise[..., :3] = np.random.normal(0, 255, (height, width, 3)).astype(np.uint8)

    # Альфа-канал: фиксированная прозрачность (0=прозрачный, 255=непрозрачный)
    alpha_value = np.clip(int(255 * alpha), 0, 255)
    noise[..., 3] = alpha_value

    return Image.fromarray(noise, mode='RGBA')


def generate_dataset(n, width, height, max_text_len, font_path, font_size_range,
                     output_path, manga_images, noise_alpha=0.3):
    ensure_directories(output_path)

    for i in tqdm(range(n), desc="Generating"):
        font_size = random.randint(*font_size_range)
        base_image = crop_random_region(manga_images, width, height).convert('RGBA')  # RGBA
        text_mask = Image.new('L', (width, height), 0)
        draw_mask = ImageDraw.Draw(text_mask)

        # Генерация текстовых элементов (только для маски)
        texts = [generate_japanese_text(max_text_len) for _ in range(3)]
        font = ImageFont.truetype(font_path, font_size)

        for text in texts:
            x = random.randint(-font_size * 2, width - font_size)
            y = random.randint(-font_size, height - font_size // 2)
            bbox = draw_mask.textbbox((x, y), text, font=font)
            draw_mask.rectangle(bbox, fill=255)

        # Генерация шума с альфа-каналом
        noise = generate_noise_mask(width, height, noise_alpha)

        # Применяем маску к альфа-каналу шума
        mask_array = np.array(text_mask)
        noise_array = np.array(noise)
        noise_alpha_channel = noise_array[:, :, 3]

        # Комбинируем маску и альфа-канал шума
        combined_alpha = np.minimum(noise_alpha_channel, mask_array)
        noise_array[:, :, 3] = combined_alpha
        noise = Image.fromarray(noise_array, mode='RGBA')

        # Наложение шума с учетом прозрачности
        final_image = Image.alpha_composite(base_image, noise)

        # Конвертация в RGB для сохранения (если нужно)
        final_image_rgb = final_image.convert('RGB')

        # Сохранение
        base_image.convert('RGB').save(os.path.join(output_path, 'Clear', f'image_{i + 1}.png'))
        text_mask.save(os.path.join(output_path, 'Mask', f'image_{i + 1}.png'))
        final_image_rgb.save(os.path.join(output_path, 'Text', f'image_{i + 1}.png'))


def main():
    parser = argparse.ArgumentParser(description="Генерация датасета манги с шумом по маске")
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--text_length", type=int, default=5)
    parser.add_argument("--font_path", type=str, default="./NotoSansJP-Bold.ttf")
    parser.add_argument("--font_size_min", type=int, default=10)
    parser.add_argument("--font_size_max", type=int, default=40)
    parser.add_argument("--path_to_save", type=str, default="./train_data")
    parser.add_argument("--manga_path", type=str, default="./clm")
    parser.add_argument("--noise_alpha", type=float, default=0.8)
    args = parser.parse_args()

    manga_images = [
        Image.open(os.path.join(args.manga_path, f)).convert('RGB')
        for f in os.listdir(args.manga_path)
    ]

    generate_dataset(
        args.count,
        args.width,
        args.height,
        args.text_length,
        args.font_path,
        (args.font_size_min, args.font_size_max),
        args.path_to_save,
        manga_images,
        args.noise_alpha
    )


if __name__ == "__main__":
    main()