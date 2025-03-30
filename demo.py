from model import DeepEraser
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import warnings
from torchvision import transforms
import matplotlib.pyplot as plt
import os

warnings.filterwarnings('ignore')


def show_images(input_images, clean_images, output_images):
    # Преобразуем все изображения в numpy массивы
    input_images = input_images.cpu().detach().numpy()
    clean_images = clean_images.cpu().detach().numpy()

    # Преобразуем список выходных изображений в numpy массив
    output_images = [output_image.squeeze(0).cpu().detach().numpy() for output_image in output_images]

    num_images = len(output_images)
    plt.figure(figsize=(15, 5))

    for i in range(num_images - 1 , num_images):
        plt.subplot(1, 3, 1)
        plt.imshow(input_images.squeeze(0), cmap='gray')  # Удаляем размерность канала
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(clean_images.squeeze(0), cmap='gray')  # Удаляем размерность канала
        plt.title("Clean Image")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(output_images[i].squeeze(0), cmap='gray')  # Удаляем размерность канала
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
    # print(torch.__version__)


    net = DeepEraser().cuda()


    reload_rec_model(net, rec_model_path)

    net.eval()

    # Задайте новый размер

    # Загрузка изображения и маски
    img = Image.open(img_path + 'input.png').convert('L')  # Убираем альфа-канал
    mask = Image.open(img_path + 'mask.png').convert('L')  # Преобразуем маску в градации серого
    print(img.size[0])
    print(img.size[1])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img = transform(img)
    mask = transform(mask)
    with torch.no_grad():

        name = 'output'

        pred_img = net(img.unsqueeze(0).cuda(), mask.unsqueeze(0).cuda(), iters=16)
        asd = 0
        print("save")
        show_images(img, mask, pred_img)
        for i in pred_img:

            i = torch.clamp(i, 0, 1)

            out = (i[0]*255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            cv2.imwrite(save_path + name + str(asd) + '.png', out[:,:,::-1])
            asd += 1



def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def main():
    rec_model_path = './deeperaser.pth'
    img_path = './input_imgs/'
    save_path =  './output_imgs/'
    rec(rec_model_path, img_path, save_path)


if __name__ == "__main__":
    main()
