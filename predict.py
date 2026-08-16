import os

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model import UNet


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = (128, 128)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "models/best_model.pth"

INPUT_IMAGE = "examples/input/cat_2.jpg"

OUTPUT_DIR = "results/predictions"

THRESHOLD = 0.5


# ============================================================
# Create output directory
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Load Model
# ============================================================

print("=" * 60)
print("U-Net Prediction")
print("=" * 60)

print(f"Device : {DEVICE}")
print(f"Model  : {MODEL_PATH}")
print(f"Input  : {INPUT_IMAGE}")

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
# Load Image
# ============================================================

image = Image.open(
    INPUT_IMAGE
).convert("RGB")

original_image = image.copy()

# Resize for model
image = image.resize(
    IMAGE_SIZE
)


# ============================================================
# Convert Image to Tensor
# ============================================================

image_array = np.array(
    image
).astype(np.float32) / 255.0

image_tensor = torch.from_numpy(
    image_array
).permute(2, 0, 1)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(
    DEVICE
)


# ============================================================
# Prediction
# ============================================================

with torch.no_grad():

    output = model(
        image_tensor
    )

    probability = torch.sigmoid(
        output
    )

    predicted_mask = (
        probability > THRESHOLD
    ).float()


# ============================================================
# Convert Prediction to NumPy
# ============================================================

predicted_mask = (
    predicted_mask
    .squeeze()
    .cpu()
    .numpy()
)


# ============================================================
# Resize Mask Back to Original Image Size
# ============================================================

original_width, original_height = (
    original_image.size
)

mask_image = Image.fromarray(
    (predicted_mask * 255).astype(
        np.uint8
    )
)

mask_image = mask_image.resize(
    (original_width, original_height),
    Image.Resampling.NEAREST
)

mask = np.array(
    mask_image
) / 255.0


# ============================================================
# Create Overlay
# ============================================================

original_array = np.array(
    original_image
)

overlay = original_array.copy()

# Apply mask to image
overlay[mask > 0.5] = (
    overlay[mask > 0.5] * 0.5
    + np.array([255, 0, 0]) * 0.5
).astype(np.uint8)


# ============================================================
# Save Results
# ============================================================

base_name = os.path.splitext(
    os.path.basename(INPUT_IMAGE)
)[0]


mask_path = os.path.join(
    OUTPUT_DIR,
    f"{base_name}_mask.png"
)

overlay_path = os.path.join(
    OUTPUT_DIR,
    f"{base_name}_overlay.png"
)

comparison_path = os.path.join(
    OUTPUT_DIR,
    f"{base_name}_prediction.png"
)


# Save mask
Image.fromarray(
    (mask * 255).astype(np.uint8)
).save(
    mask_path
)


# Save overlay
Image.fromarray(
    overlay
).save(
    overlay_path
)


# ============================================================
# Create Comparison Figure
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


# Original image
axes[0].imshow(
    original_image
)

axes[0].set_title(
    "Original Image"
)

axes[0].axis("off")


# Predicted mask
axes[1].imshow(
    mask,
    cmap="gray"
)

axes[1].set_title(
    "Predicted Mask"
)

axes[1].axis("off")


# Overlay
axes[2].imshow(
    overlay
)

axes[2].set_title(
    "Segmentation Overlay"
)

axes[2].axis("off")


plt.tight_layout()

plt.savefig(
    comparison_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Final Output
# ============================================================

print()
print("=" * 60)
print("Prediction completed!")
print("=" * 60)

print(
    f"Mask saved    : {mask_path}"
)

print(
    f"Overlay saved : {overlay_path}"
)

print(
    f"Comparison    : {comparison_path}"
)

print("=" * 60)