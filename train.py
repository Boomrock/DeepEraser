import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from model import DeepEraser
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import keyboard
import pandas as pd

warnings.filterwarnings("ignore")
os.system("cls")


class EraserDataset(Dataset):
    def __init__(self, img_dir, mask_dir, clean_dir, transform=None, start_items=0, count_items=1000):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.image_filenames = os.listdir(img_dir)[start_items : start_items + count_items]

        self.images = []
        self.clean_images = []
        self.masks = []
        print("Загрузка тренировочных данных")
        for file in tqdm(self.image_filenames):
            image = Image.open(os.path.join(self.img_dir, file)).convert("RGB")
            mask = Image.open(os.path.join(self.mask_dir, file)).convert("L")
            clean_image = Image.open(os.path.join(self.clean_dir, file)).convert("RGB")

            if self.transform:
                image = self.transform(image)
                clean_image = self.transform(clean_image)
                mask = self.transform(mask)

            self.images.append(image)
            self.clean_images.append(clean_image)
            self.masks.append(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.clean_images[idx], self.masks[idx]


def reload_rec_model(model, path=""):
    if not path:
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location="cuda:0")
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model


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


def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    pbar = tqdm(range(num_epochs), desc="Training")
    metadata = pd.DataFrame({"epoch": [], "loss": []})
    try:
        for epoch in pbar:
            epoch_loss = 0
            batch_counter = 1
            for images, clean_images, masks in tqdm(train_loader, leave=False):
                images, masks, clean_images = images.to(device), masks.to(device), clean_images.to(device)

                optimizer.zero_grad()

                outputs = model(images, masks, iters=8)
                outputs = outputs[-1]
                loss = criterion(outputs, clean_images)
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()
                pbar.set_postfix({"Epoch Loss": f"{epoch_loss / batch_counter:.6f}", "Local Loss": f"{loss:.6f}"})
                batch_counter += 1

                if keyboard.is_pressed("v"):
                    show_images(images, clean_images, outputs)
            metadata.loc[epoch] = {"epoch": epoch, "loss": epoch_loss / len(train_loader)}
    except KeyboardInterrupt as _:
        print("Interrupted, returning not completed model")
    except Exception as e:
        print(f"Exception occured: {e}")
    finally:
        return model, metadata


def main():
    parser = argparse.ArgumentParser(description="Training DeepEraser model.")
    parser.add_argument("--img_dir", type=str, default="./train_data/Text", help="Path to text images. (default: ./train_data/Text)")
    parser.add_argument("--mask_dir", type=str, default="./train_data/Mask", help="Path to mask images. (default: ./train_data/Mask)")
    parser.add_argument("--clean_dir", type=str, default="./train_data/Clear", help="Path to clean images. (default: ./train_data/Clear)")
    parser.add_argument("--save_model_path", type=str, default="./deeperaser1.pth", help="Path to save the model. (default: ./deeperaser1.pth)")
    parser.add_argument("--rec_model_path", type=str, default="./deeperaser.pth", help="Path to a pretrained model. (default: ./deeperaser.pth)")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size for training. (default: 6)")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate. (default: 5e-6)")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of training epochs. (default: 4)")
    parser.add_argument("--start_items", type=int, default=0, help="Starting index for dataset. (default: 0)")
    parser.add_argument("--count_items", type=int, default=1000, help="Number of items in the dataset. (default: 1000)")
    parser.add_argument("--metadata_path", type=str, default="./metadata.csv", help="training metadata file path (default: ./metadata.csv)")

    args = parser.parse_args()

    print("Parameters:")
    print(f"Image Directory: {args.img_dir}")
    print(f"Mask Directory: {args.mask_dir}")
    print(f"Clean Directory: {args.clean_dir}")
    print(f"Save Model Path: {args.save_model_path}")
    print(f"Pretrained Model Path: {args.rec_model_path}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Number of Epochs: {args.num_epochs}")
    print(f"Starting Index: {args.start_items}")
    print(f"Count of Items: {args.count_items}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),  # Изменение размера на 32x32
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_dataset = EraserDataset(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        clean_dir=args.clean_dir,
        transform=transform,
        start_items=args.start_items,
        count_items=args.count_items,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = DeepEraser().to(device)
    model = nn.DataParallel(model)
    model = reload_rec_model(model, args.rec_model_path)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    trained_model, metadata = train_model(model, train_loader, criterion, optimizer, args.num_epochs, device)
    metadata.to_csv(f"{args.metadata_path}")
    torch.save(trained_model.state_dict(), args.save_model_path)


if __name__ == "__main__":
    main()
