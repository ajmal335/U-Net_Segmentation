import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import PetSegmentationDataset
from model import UNet


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = (128, 128)

# Start small on CPU to verify the complete pipeline.
# Set to None later to use the complete dataset.
TRAIN_LIMIT = 500
VAL_LIMIT = 200

BATCH_SIZE = 8
NUM_EPOCHS = 5

LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_WORKERS = 0


# ============================================================
# Paths
# ============================================================

TRAIN_IMAGE_DIR = "dataset/train/images"
TRAIN_MASK_DIR = "dataset/train/masks"

VAL_IMAGE_DIR = "dataset/val/images"
VAL_MASK_DIR = "dataset/val/masks"

MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pth"
)


# ============================================================
# Dice Loss
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()

        self.smooth = smooth

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()

        dice = (
            (2.0 * intersection + self.smooth)
            /
            (
                predictions.sum()
                + targets.sum()
                + self.smooth
            )
        )

        return 1.0 - dice


# ============================================================
# BCE + Dice Loss
# ============================================================

class BCEDiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, predictions, targets):

        bce_loss = self.bce(
            predictions,
            targets
        )

        dice_loss = self.dice(
            predictions,
            targets
        )

        return bce_loss + dice_loss


# ============================================================
# Dice Score
# ============================================================

def dice_score(predictions, targets, threshold=0.5):

    predictions = torch.sigmoid(predictions)

    predictions = (
        predictions > threshold
    ).float()

    intersection = (
        predictions * targets
    ).sum()

    dice = (
        2.0 * intersection + 1.0
    ) / (
        predictions.sum()
        + targets.sum()
        + 1.0
    )

    return dice.item()


# ============================================================
# Dataset
# ============================================================

full_train_dataset = PetSegmentationDataset(
    image_dir=TRAIN_IMAGE_DIR,
    mask_dir=TRAIN_MASK_DIR,
    image_size=IMAGE_SIZE
)

full_val_dataset = PetSegmentationDataset(
    image_dir=VAL_IMAGE_DIR,
    mask_dir=VAL_MASK_DIR,
    image_size=IMAGE_SIZE
)


# ------------------------------------------------------------
# Use a small subset initially
# ------------------------------------------------------------

if TRAIN_LIMIT is not None:
    train_dataset = Subset(
        full_train_dataset,
        range(
            min(
                TRAIN_LIMIT,
                len(full_train_dataset)
            )
        )
    )
else:
    train_dataset = full_train_dataset


if VAL_LIMIT is not None:
    val_dataset = Subset(
        full_val_dataset,
        range(
            min(
                VAL_LIMIT,
                len(full_val_dataset)
            )
        )
    )
else:
    val_dataset = full_val_dataset


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# Information
# ============================================================

print("=" * 60)
print("U-Net Training")
print("=" * 60)

print(f"Image size        : {IMAGE_SIZE}")
print(f"Training images   : {len(train_dataset)}")
print(f"Validation images : {len(val_dataset)}")
print(f"Batch size        : {BATCH_SIZE}")
print(f"Epochs            : {NUM_EPOCHS}")
print(f"Learning rate     : {LEARNING_RATE}")
print(f"Device            : {DEVICE}")

if TRAIN_LIMIT is not None:
    print(f"Training limit    : {TRAIN_LIMIT}")

if VAL_LIMIT is not None:
    print(f"Validation limit  : {VAL_LIMIT}")

print("=" * 60)


# ============================================================
# Model
# ============================================================

# Your existing UNet accepts in_channels only.
model = UNet(
    in_channels=3
).to(DEVICE)


# ============================================================
# Loss and Optimizer
# ============================================================

criterion = BCEDiceLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Model directory
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# Training
# ============================================================

best_val_dice = 0.0


for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------------
    # Training mode
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_dice = 0.0

    train_progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} - Training"
    )

    for images, masks in train_progress:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Forward pass
        predictions = model(images)

        # Calculate loss
        loss = criterion(
            predictions,
            masks
        )

        # Clear previous gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Track metrics
        train_loss += loss.item()

        train_dice += dice_score(
            predictions,
            masks
        )

        train_progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    train_loss /= len(train_loader)
    train_dice /= len(train_loader)


    # --------------------------------------------------------
    # Validation mode
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_dice = 0.0

    val_progress = tqdm(
        val_loader,
        desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} - Validation"
    )

    with torch.no_grad():

        for images, masks in val_progress:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)

            loss = criterion(
                predictions,
                masks
            )

            val_loss += loss.item()

            val_dice += dice_score(
                predictions,
                masks
            )

            val_progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )


    val_loss /= len(val_loader)
    val_dice /= len(val_loader)


    # --------------------------------------------------------
    # Epoch results
    # --------------------------------------------------------

    print()
    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Dice: {train_dice:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_dice > best_val_dice:

        best_val_dice = val_dice

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            f"✓ Best model saved "
            f"(Val Dice: {val_dice:.4f})"
        )


# ============================================================
# Training complete
# ============================================================

print()
print("=" * 60)
print("Training completed!")
print("=" * 60)

print(
    f"Best validation Dice: "
    f"{best_val_dice:.4f}"
)

print(
    f"Best model saved to: "
    f"{BEST_MODEL_PATH}"
)