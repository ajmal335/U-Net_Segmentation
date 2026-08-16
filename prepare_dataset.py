from pathlib import Path
import random
import shutil
from PIL import Image


# =========================
# Paths
# =========================

SOURCE_IMAGES = Path("dataset_download/images")
SOURCE_MASKS = Path("dataset_download/annotations/trimaps")

OUTPUT_DIR = Path("dataset")


# =========================
# Dataset split
# =========================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# =========================
# Get images
# =========================

images = sorted(SOURCE_IMAGES.glob("*.jpg"))

print(f"Total images found: {len(images)}")


# =========================
# Shuffle
# =========================

random.seed(RANDOM_SEED)
random.shuffle(images)


# =========================
# Split dataset
# =========================

total = len(images)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]


print(f"Train images: {len(train_images)}")
print(f"Val images:   {len(val_images)}")
print(f"Test images:  {len(test_images)}")


# =========================
# Convert trimap to binary mask
# =========================

def create_binary_mask(source_mask, destination_mask):
    """
    Convert Oxford-IIIT Pet trimap into a binary mask.

    Trimap:
        1 = foreground/pet
        2 = background
        3 = boundary

    Binary:
        255 = pet
        0   = background
    """

    mask = Image.open(source_mask)

    binary_mask = mask.point(
        lambda pixel: 255 if pixel == 1 else 0
    )

    binary_mask.save(destination_mask)


# =========================
# Process split
# =========================

def process_split(image_list, split_name):

    image_output_dir = OUTPUT_DIR / split_name / "images"
    mask_output_dir = OUTPUT_DIR / split_name / "masks"

    image_output_dir.mkdir(parents=True, exist_ok=True)
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_list:

        image_name = image_path.stem

        # Copy image
        destination_image = image_output_dir / image_path.name

        shutil.copy2(
            image_path,
            destination_image
        )

        # Find corresponding trimap
        mask_path = SOURCE_MASKS / f"{image_name}.png"

        if not mask_path.exists():
            print(f"WARNING: Mask not found for {image_name}")
            continue

        # Create binary mask
        destination_mask = mask_output_dir / f"{image_name}.png"

        create_binary_mask(
            mask_path,
            destination_mask
        )

    print(f"{split_name} completed.")


# =========================
# Run processing
# =========================

if __name__ == "__main__":

    process_split(train_images, "train")
    process_split(val_images, "val")
    process_split(test_images, "test")

    print()
    print("=" * 40)
    print("Dataset preparation completed!")
    print("=" * 40)