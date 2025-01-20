from model import DeepEraser
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import warnings


warnings.filterwarnings('ignore')


def reload_rec_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


def rec(rec_model_path, img_path, save_path):
    # print(torch.__version__)


    net = DeepEraser().cuda()


    reload_rec_model(net, rec_model_path)

    net.eval()

    # Задайте новый размер
    new_size = (600, 800)  # Замените width и height на нужные значения

    # Загрузка изображения и маски
    img = Image.open(img_path + 'input.png').convert('RGB')  # Убираем альфа-канал
    mask = Image.open(img_path + 'mask.png').convert('L')  # Преобразуем маску в градации серого

    # Уменьшение размера
    img = img.resize(new_size, Image.LANCZOS)  # Изменяем размер изображения
    mask = mask.resize(new_size, Image.LANCZOS)  # Изменяем размер маски (используем NEAREST для маски)

    # Преобразование в numpy массивы
    img = np.array(img)[:, :, :3]  # Убедитесь, что только RGB каналы
    mask = np.array(mask)[:, :]  # Маска без изменений

    # Преобразование в тензоры PyTorch
    im = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
    mask = torch.from_numpy(mask / 255.0).unsqueeze(0).float()

    with torch.no_grad():

        name = 'output'

        pred_img = net(im.unsqueeze(0).cuda(), mask.unsqueeze(0).cuda())
        asd = 0
        print("save")
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
    rec_model_path = './deeperaser1.pth'
    img_path = './input_imgs/'
    save_path =  './output_imgs/'
    rec(rec_model_path, img_path, save_path)


if __name__ == "__main__":
    main()
