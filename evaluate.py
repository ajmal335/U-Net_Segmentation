import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import PetSegmentationDataset
from model import UNet


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = (128, 128)

BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/best_model.pth"

TEST_IMAGE_DIR = "dataset/test/images"
TEST_MASK_DIR = "dataset/test/masks"

RESULTS_DIR = "results"

NUM_VISUALIZATIONS = 10


# ============================================================
# Create results directory
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


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
# Load Test Dataset
# ============================================================

test_dataset = PetSegmentationDataset(
    image_dir=TEST_IMAGE_DIR,
    mask_dir=TEST_MASK_DIR,
    image_size=IMAGE_SIZE
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Information
# ============================================================

print("=" * 60)
print("U-Net Test Evaluation")
print("=" * 60)

print(f"Test images : {len(test_dataset)}")
print(f"Image size  : {IMAGE_SIZE}")
print(f"Batch size  : {BATCH_SIZE}")
print(f"Device      : {DEVICE}")
print(f"Model       : {MODEL_PATH}")

print("=" * 60)


# ============================================================
# Load Model
# ============================================================

model = UNet(
    in_channels=3
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()


# ============================================================
# Test Evaluation
# ============================================================

total_dice = 0.0
num_batches = 0

visualizations_saved = 0


with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Model prediction
        predictions = model(images)

        # Dice score
        batch_dice = dice_score(
            predictions,
            masks
        )

        total_dice += batch_dice
        num_batches += 1

        # ----------------------------------------------------
        # Save visualizations
        # ----------------------------------------------------

        probabilities = torch.sigmoid(
            predictions
        )

        predicted_masks = (
            probabilities > 0.5
        ).float()


        for i in range(images.size(0)):

            if visualizations_saved >= NUM_VISUALIZATIONS:
                break

            # Original image
            image = images[i].cpu()

            image = image.permute(
                1, 2, 0
            ).numpy()

            # Ground-truth mask
            ground_truth = masks[i].cpu().squeeze().numpy()

            # Predicted mask
            predicted = (
                predicted_masks[i]
                .cpu()
                .squeeze()
                .numpy()
            )

            # ------------------------------------------------
            # Create figure
            # ------------------------------------------------

            fig, axes = plt.subplots(
                1,
                4,
                figsize=(16, 4)
            )

            # Original
            axes[0].imshow(image)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            # Ground truth
            axes[1].imshow(
                ground_truth,
                cmap="gray"
            )
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            # Prediction
            axes[2].imshow(
                predicted,
                cmap="gray"
            )
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            # Overlay
            axes[3].imshow(image)

            axes[3].imshow(
                predicted,
                cmap="jet",
                alpha=0.35
            )

            axes[3].set_title("Prediction Overlay")
            axes[3].axis("off")

            plt.tight_layout()

            output_path = os.path.join(
                RESULTS_DIR,
                f"segmentation_{visualizations_saved + 1}.png"
            )

            plt.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight"
            )

            plt.close()

            visualizations_saved += 1


# ============================================================
# Final Dice Score
# ============================================================

test_dice = (
    total_dice / num_batches
)


# ============================================================
# Save Metrics
# ============================================================

metrics_path = os.path.join(
    RESULTS_DIR,
    "metrics.txt"
)

with open(
    metrics_path,
    "w"
) as file:

    file.write(
        "U-Net Test Evaluation\n"
    )

    file.write(
        "=====================\n"
    )

    file.write(
        f"Test Images : {len(test_dataset)}\n"
    )

    file.write(
        f"Image Size  : {IMAGE_SIZE}\n"
    )

    file.write(
        f"Test Dice   : {test_dice:.4f}\n"
    )


# ============================================================
# Print Results
# ============================================================

print()
print("=" * 60)
print("Test Evaluation Completed!")
print("=" * 60)

print(
    f"Test Dice Score : {test_dice:.4f}"
)

print(
    f"Visualizations  : {visualizations_saved}"
)

print(
    f"Results saved to: {RESULTS_DIR}/"
)

print(
    f"Metrics saved to : {metrics_path}"
)

print("=" * 60)