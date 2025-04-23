import os
import shutil


def restructure_dataset(base_dir="data"):
    label_folders = {"class0": "non_cancerous", "class1": "cancerous"}

    # Create target folders
    for folder in label_folders.values():
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

    moved = 0

    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".png"):
                if "class0" in file:
                    label = "class0"
                elif "class1" in file:
                    label = "class1"
                else:
                    print(f"⚠️ Skipping file with unlabelled name: {file}")
                    continue

                src_path = os.path.join(root, file)
                dst_path = os.path.join(base_dir, label_folders[label], file)

                if not os.path.exists(dst_path):
                    shutil.move(src_path, dst_path)
                    moved += 1

    print(f"✅ Restructuring complete — moved {moved} images.")


if __name__ == "__main__":
    restructure_dataset()
