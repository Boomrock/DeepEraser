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

warnings.filterwarnings('ignore')


class EraserDataset(Dataset):
    def __init__(self, img_dir, mask_dir, clean_dir, transform=None, start_items=0, count_items=1000):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.image_filenames = os.listdir(img_dir)[start_items:start_items + count_items]

        self.images = []
        self.clean_images = []
        self.masks = []

        for file in self.image_filenames:
            image = Image.open(os.path.join(self.img_dir, file)).convert("RGB")
            mask = Image.open(os.path.join(self.mask_dir, file)).convert("L")
            clean_image = Image.open(os.path.join(self.clean_dir, file)).convert("RGB")

            if self.transform:
                image = self.transform(image)
                clean_image = self.transform(clean_image)
                mask = transforms.ToTensor()(mask)

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
        pretrained_dict = torch.load(path, map_location='cuda:0')
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model


def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    pbar = tqdm(range(num_epochs), desc="Training")

    for epoch in pbar:
        epoch_loss = 0
        for images, clean_images, masks in tqdm(train_loader, leave=False):
            images, masks, clean_images = images.to(device), masks.to(device), clean_images.to(device)

            optimizer.zero_grad()

            outputs = model(images, masks, iters=epoch % 4 + 1 + 4)

            loss = criterion(outputs[-1], clean_images)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            pbar.set_postfix(
                {
                    'Epoch Loss': f"{epoch_loss / len(train_loader):.6f}",
                    'Local Loss': f"{loss:.6f}"
                })



    print("Training completed.")
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
        plt.axis('off')

        plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(clean_images[i].transpose(1, 2, 0))
        plt.title("Clean Image")
        plt.axis('off')

        plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(output_images[i].transpose(1, 2, 0))
        plt.title("Output Image")
        plt.axis('off')

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Training DeepEraser model.")
    parser.add_argument('--img_dir', type=str, default='./train_data/Text', help='Path to text images.')
    parser.add_argument('--mask_dir', type=str, default='./train_data/Mask', help='Path to mask images.')
    parser.add_argument('--clean_dir', type=str, default='./train_data/Clear', help='Path to clean images.')
    parser.add_argument('--save_model_path', type=str, default='./deeperaser1.pth', help='Path to save the model.')
    parser.add_argument('--rec_model_path', type=str, default='./deeperaser.pth', help='Path to a pretrained model.')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=4, help='Number of training epochs.')
    parser.add_argument('--start_items', type=int, default=0, help='Starting index for dataset.')
    parser.add_argument('--count_items', type=int, default=1000, help='Number of items in the dataset.')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = EraserDataset(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        clean_dir=args.clean_dir,
        transform=transform,
        start_items=args.start_items,
        count_items=args.count_items
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = DeepEraser().to(device)
    model = reload_rec_model(model, args.rec_model_path)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    trained_model = train_model(model, train_loader, criterion, optimizer, args.num_epochs, device)

    trained_model = nn.DataParallel(trained_model)
    torch.save(trained_model.state_dict(), args.save_model_path)


if __name__ == "__main__":
    main()
