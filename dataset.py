import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class PetSegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, image_size=(224, 224)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(
            self.mask_dir,
            os.path.splitext(image_name)[0] + ".png"
        )

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load segmentation mask
        mask = Image.open(mask_path).convert("L")

        # Resize image
        image = TF.resize(
            image,
            self.image_size,
            antialias=True
        )

        # Resize mask using NEAREST interpolation
        # Important: masks contain class labels, so we
        # should NOT use normal/bilinear interpolation.
        mask = TF.resize(
            mask,
            self.image_size,
            interpolation=TF.InterpolationMode.NEAREST
        )

        # Convert image to tensor
        image = TF.to_tensor(image)

        # Convert mask to tensor
        mask = TF.to_tensor(mask)

        # Oxford-IIIT Pet trimap:
        # 1 = foreground
        # 2 = background
        # 3 = border
        #
        # For this binary segmentation project:
        # foreground -> 1
        # everything else -> 0

        mask = (mask == 1).float()

        return image, mask


if __name__ == "__main__":

    dataset = PetSegmentationDataset(
        image_dir="dataset/train/images",
        mask_dir="dataset/train/masks"
    )

    print("Dataset size:", len(dataset))

    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape :", mask.shape)

    print("Image min:", image.min().item())
    print("Image max:", image.max().item())

    print("Mask unique values:", torch.unique(mask))