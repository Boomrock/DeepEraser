from model import DeepEraser
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import warnings
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from generator_train_data import generate_noise_mask

warnings.filterwarnings('ignore')

def show_images(input_images, clean_images, output_images):
    input_images = input_images.squeeze().cpu().detach().numpy()
    clean_images = clean_images.squeeze().cpu().detach().numpy()
    output_images = [img.squeeze().cpu().detach().numpy() for img in output_images]

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(input_images.transpose(1, 2, 0))
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(132)
    plt.imshow(clean_images.transpose(1, 2, 0))
    plt.title("Clean Image")
    plt.axis("off")

    plt.subplot(133)
    plt.imshow(output_images[-1].transpose(1, 2, 0))
    plt.title("Output Image")
    plt.axis("off")
    plt.show()

def reload_rec_model(model, path=""):
    if not path or not os.path.exists(path):
        return model
    checkpoint = torch.load(path, map_location="cuda:0")
    state_dict = checkpoint['model_state_dict']
    # Убираем префикс 'module.' если модель обучалась с DataParallel

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model
def rec(rec_model_path, img_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    net = DeepEraser().cuda()
    net = reload_rec_model(net, rec_model_path)
    net.eval()

    # Загрузка и масштабирование изображений
    img = Image.open(os.path.join(img_path, 'input.png')).convert('RGBA')
    mask = Image.open(os.path.join(img_path, 'mask.png')).convert('L')

    # Масштабирование до 64x64
    img_resized = img.resize((64, 64))
    mask_resized = mask.resize((64, 64))

    # Генерация шума нужного размера
    noise = generate_noise_mask(64, 64, alpha=1)

    # Обработка альфа-канала шума по маске
    mask_array = np.array(mask_resized)
    noise_array = np.array(noise)
    noise_alpha_channel = noise_array[:, :, 3]
    combined_alpha = np.minimum(noise_alpha_channel, mask_array)
    noise_array[:, :, 3] = combined_alpha
    noise_image = Image.fromarray(noise_array, mode='RGBA')

    # Наложение шума через альфа-композит
    final_image = Image.alpha_composite(img_resized, noise_image).convert('RGB')

    # Преобразования для нейросети
    transform = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.clamp(0, 1)
    ])

    mask_tensor = transform(mask_resized).round()
    img_tensor = transform(final_image)

    with torch.no_grad():
        pred_imgs = net(
            img_tensor.unsqueeze(0).cuda(),
            mask_tensor.unsqueeze(0).cuda())

        for i, pred in enumerate(pred_imgs):
            pred = torch.clamp(pred, 0, 1)
            out = (pred[0].cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(out).save(os.path.join(save_path, f'output_{i}.png'))

    show_images(img_tensor, transform(img_resized), pred_imgs.cpu())

def main():
    rec(
        rec_model_path='./deeperaser.pth',
        img_path='./input_imgs/',
        save_path='./output_imgs/'
    )

if __name__ == "__main__":
    main()