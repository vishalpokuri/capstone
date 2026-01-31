import os
import shutil
from collections import Counter

# -----------------------------
# CONFIGURATION
# -----------------------------

YOLO_DATASET_PATH = "dataset"          # path to original YOLO dataset
OUTPUT_DATASET_PATH = "clean_dataset"  # path to new classification dataset

# Class ID → Class Name mapping (from data.yaml)
CLASS_MAP = {
    0: "-K",
    1: "-N",
    2: "-P",
    3: "FN"
}

SPLITS = {
    "train": "train",
    "val": "valid",
    "test": "test"
}

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def ensure_dir(path):
    """Create directory if it does not exist"""
    os.makedirs(path, exist_ok=True)

# -----------------------------
# MAIN CONVERSION LOGIC
# -----------------------------

def convert_yolo_to_classification():
    print("🚀 Starting YOLO → Classification conversion...\n")

    overall_counter = Counter()

    for split_name, split_folder in SPLITS.items():
        print(f"📂 Processing split: {split_name}")

        images_dir = os.path.join(YOLO_DATASET_PATH, split_folder, "images")
        labels_dir = os.path.join(YOLO_DATASET_PATH, split_folder, "labels")

        # Create output directories
        for class_name in CLASS_MAP.values():
            ensure_dir(os.path.join(OUTPUT_DATASET_PATH, split_name, class_name))

        split_counter = Counter()

        for label_file in os.listdir(labels_dir):
            if not label_file.endswith(".txt"):
                continue

            label_path = os.path.join(labels_dir, label_file)

            # Read first class ID from label file
            with open(label_path, "r") as f:
                line = f.readline().strip()

            if not line:
                continue  # skip empty label files

            class_id = int(line.split()[0])
            class_name = CLASS_MAP.get(class_id)

            if class_name is None:
                continue

            # Image filename
            image_name = label_file.replace(".txt", ".jpg")
            image_path = os.path.join(images_dir, image_name)

            if not os.path.exists(image_path):
                # Try png if jpg not found
                image_name = label_file.replace(".txt", ".png")
                image_path = os.path.join(images_dir, image_name)

            if not os.path.exists(image_path):
                print(f"⚠️ Image not found for label: {label_file}")
                continue

            # Destination path
            dest_path = os.path.join(
                OUTPUT_DATASET_PATH,
                split_name,
                class_name,
                image_name
            )

            shutil.copy(image_path, dest_path)

            split_counter[class_name] += 1
            overall_counter[class_name] += 1

        print(f"✅ {split_name} distribution: {dict(split_counter)}\n")

    print("🎉 Conversion complete!")
    print("📊 Overall class distribution:")
    for cls, count in overall_counter.items():
        print(f"  {cls}: {count}")

# -----------------------------
# RUN SCRIPT
# -----------------------------

if __name__ == "__main__":
    convert_yolo_to_classification()
