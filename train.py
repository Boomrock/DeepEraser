import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from model import DeepEraser
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import keyboard
import pandas as pd
import random  # Добавлен импорт

warnings.filterwarnings("ignore")
os.system("cls")


class EraserDataset(Dataset):
    def __init__(self, img_dir, mask_dir, clean_dir, count, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.clean_dir = clean_dir
        self.transform = transform

        self.image_filenames = sorted(os.listdir(img_dir))
        self.clean_filenames = sorted(os.listdir(clean_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

        self.image_filenames = self.image_filenames[:count]
        self.clean_filenames = self.clean_filenames[:count]
        self.mask_filenames = self.mask_filenames[:count]
        # Проверка соответствия файлов
        assert len(self.image_filenames) == len(self.clean_filenames) == len(
            self.mask_filenames), "Mismatch in dataset sizes"
        for img, msk, cln in zip(self.image_filenames, self.mask_filenames, self.clean_filenames):
            assert img == msk == cln, f"Filename mismatch: {img}, {msk}, {cln}"

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]

        image = Image.open(os.path.join(self.img_dir, img_name)).convert("L")
        mask = Image.open(os.path.join(self.mask_dir, img_name)).convert("L")
        clean = Image.open(os.path.join(self.clean_dir, img_name)).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            clean = self.transform(clean)

        return image, clean, mask


def get_data_loaders(
        img_dir: str,
        mask_dir: str,
        clean_dir: str,
        count: int,
        batch_size: int = 6,
        val_split: float = 0.2,
        transform=None,
        seed: int = 42
) -> (DataLoader, DataLoader):
    """Создает загрузчики данных с автоматическим разделением"""

    full_dataset = EraserDataset(img_dir, mask_dir, clean_dir, count, transform=transform)

    # Рассчитываем размеры выборок
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    print(train_size)
    # Фиксируем генератор случайных чисел для воспроизводимости
    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    print(f"Размер обучающей выборки: {len(train_dataset)}")
    print(f"Размер валидационной выборки: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def reload_rec_model(model, optimizer, path=""):
    if not path or not os.path.exists(path):
        return model, optimizer
    checkpoint = torch.load(path, map_location="cuda:0")
    state_dict = checkpoint['model_state_dict']
    # Убираем префикс 'module.' если модель обучалась с DataParallel

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer
def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, clean_images, masks in val_loader:
            images, masks, clean_images = images.to(device), masks.to(device), clean_images.to(device)
            outputs = model(images, masks, iters=8)[-1]
            loss = criterion(outputs, clean_images)
            val_loss += loss.item()
    return val_loss / len(val_loader)
def show_images(input_images, clean_images, output_images):
    input_images = input_images.cpu().detach().numpy()
    clean_images = clean_images.cpu().detach().numpy()
    output_images = output_images.cpu().detach().numpy()

    num_images = min(5, input_images.shape[0])
    plt.figure(figsize=(15, 5))

    for i in range(num_images):
        plt.subplot(3, num_images, i + 1)
        plt.imshow(input_images[i].transpose(1, 2, 0))
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(clean_images[i].transpose(1, 2, 0))
        plt.title("Clean Image")
        plt.axis("off")

        plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(output_images[i].transpose(1, 2, 0))
        plt.title("Output Image")
        plt.axis("off")

    plt.show()


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, metadata_path):

    pbar = tqdm(range(num_epochs), desc="Training")
    best_val_loss = float('inf')
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
        if metadata.empty:
            metadata = pd.DataFrame({"epoch": [], "train_loss": [], "val_loss": []})
    else:
        metadata = pd.DataFrame({"epoch": [], "train_loss": [], "val_loss": []})

    for epoch in pbar:
        batch_counter = 1

        # Training phase
        model.train()
        train_epoch_loss = 0
        for images, clean_images, masks in tqdm(train_loader, leave=False, desc="Train"):
            images, masks, clean_images = images.to(device), masks.to(device), clean_images.to(device)
            optimizer.zero_grad()
            outputs = model(images, masks, iters=8)[-1]
            loss = criterion(outputs, clean_images)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
            pbar.set_postfix({"Epoch Loss": f"{train_epoch_loss / batch_counter:.6f}", "Local Loss": f"{loss:.6f}"})
            batch_counter += 1
            if keyboard.is_pressed("v"):
                show_images(images, clean_images, outputs)
        # Validation phase
        val_loss = 0
        if val_loader is not None:
            val_loss = validate_model(model, val_loader, criterion, device)
            print("vall_loss" + str(val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.module.state_dict(), f"best_model_epoch_{epoch}.pth")

        # Logging
        avg_train_loss = train_epoch_loss / len(train_loader)

        last_row = 0

        if not metadata.empty:
            last_row = metadata.iloc[-1]['epoch'] + 1
        metadata.loc[last_row] = {
            "epoch": last_row,
            "train_loss": avg_train_loss,
            "val_loss": val_loss
        }
    try:
        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.6f}, Val Loss {val_loss:.6f}")
    except KeyboardInterrupt:
        print("Interrupted, returning not completed model")
        return model, metadata
    finally:
        return model, metadata


def main():
    parser = argparse.ArgumentParser(description="Training DeepEraser model.")
    parser.add_argument("--img_dir", type=str, default="./train_data/Text",
                        help="Path to text images (default: ./train_data/Text)")
    parser.add_argument("--mask_dir", type=str, default="./train_data/Mask",
                        help="Path to mask images (default: ./train_data/Mask)")
    parser.add_argument("--clean_dir", type=str, default="./train_data/Clear",
                        help="Path to clean images (default: ./train_data/Clear)")
    parser.add_argument("--save_model_path", type=str, default="./deeperaser.pth",
                        help="Path to save the model (default: ./deeperaser.pth)")
    parser.add_argument("--rec_model_path", type=str, default="./deeperaser.pth",
                    help="Path to save the model (default: ./deeperaser.pth)")
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Batch size for training (default: 6)")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate (default: 5e-6)")
    parser.add_argument("--num_epochs", type=int, default=4,
                        help="Number of training epochs (default: 4)")
    parser.add_argument("--count", type=int, default=1000,
                        help="count of image (default: 1000)")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="Validation split proportion (default: 0.2)")
    parser.add_argument("--metadata_path", type=str, default="./metadata.csv",
                        help="Training metadata file path (default: ./metadata.csv)")
    args = parser.parse_args()
    print("Parameters:")
    print(f"Image Directory: {args.img_dir}")
    print(f"Mask Directory: {args.mask_dir}")
    print(f"Clean Directory: {args.clean_dir}")
    print(f"Save Model Path: {args.save_model_path}")
    print(f"Pretrained Model Path: {args.rec_model_path}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"count of images: {args.count}")
    print(f"Number of Epochs: {args.num_epochs}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Создаем загрузчики данных
    train_loader, val_loader = get_data_loaders(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        clean_dir=args.clean_dir,
        count=args.count,
        batch_size=args.batch_size,
        val_split=args.validation_split,
        transform=transform
    )

    model = DeepEraser().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model, optimizer = reload_rec_model(model, optimizer, args.rec_model_path)
    model = nn.DataParallel(model)

    # Обучение с валидацией
    trained_model, metadata = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        metadata_path = args.metadata_path
    )

    # Сохранение результатов
    metadata.to_csv(args.metadata_path)
    torch.save({
        'model_state_dict': trained_model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, args.save_model_path)


if __name__ == "__main__":
    main()