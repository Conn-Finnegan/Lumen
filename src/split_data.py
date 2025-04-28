import os
import shutil
import random


def split_data(source_dir, output_dir, split_ratio=0.8):
    categories = ["cancerous", "non_cancerous"]

    for category in categories:
        src_folder = os.path.join(source_dir, category)
        train_folder = os.path.join(output_dir, "train", category)
        val_folder = os.path.join(output_dir, "val", category)

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        images = [
            f
            for f in os.listdir(src_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        random.shuffle(images)

        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Copy files
        for img in train_images:
            src_path = os.path.join(src_folder, img)
            dst_path = os.path.join(train_folder, img)
            shutil.copy2(src_path, dst_path)

        for img in val_images:
            src_path = os.path.join(src_folder, img)
            dst_path = os.path.join(val_folder, img)
            shutil.copy2(src_path, dst_path)

        print(
            f"✅ {category}: {len(train_images)} train images, {len(val_images)} val images"
        )


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, "data")
    split_data(data_dir, data_dir, split_ratio=0.8)
