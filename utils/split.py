import os
import shutil
import random

def split_dataset(
    base_dir,
    output_dir,
    train_ratio=0.8,
    seed=42
):
    random.seed(seed)
    
    tiles_dir = os.path.join(base_dir, "Tiles")
    masks_dir = os.path.join(base_dir, "Mask")

    image_names = sorted(os.listdir(tiles_dir))
    total_images = len(image_names)

    random.shuffle(image_names)
    split_index = int(train_ratio * total_images)
    
    train_images = image_names[:split_index]
    test_images = image_names[split_index:]

    for split in ['train', 'test']:
        for subfolder in ['Tiles', 'Mask']:
            os.makedirs(os.path.join(output_dir, split, subfolder), exist_ok=True)

    def copy_files(split_name, image_list):
        for name in image_list:
            tile_src = os.path.join(tiles_dir, name)
            mask_src = os.path.join(masks_dir, name)

            tile_dst = os.path.join(output_dir, split_name, 'Tiles', name)
            mask_dst = os.path.join(output_dir, split_name, 'Mask', name)

            if os.path.exists(tile_src) and os.path.exists(mask_src):
                shutil.copy2(tile_src, tile_dst)
                shutil.copy2(mask_src, mask_dst)
            else:
                print(f"Warning: Missing file for {name}, skipping...")

    copy_files("train", train_images)
    copy_files("test", test_images)

    print(f"Split complete. {len(train_images)} training and {len(test_images)} testing samples saved to '{output_dir}'.")

if __name__ == "__main__":
    split_dataset(
        base_dir="Data\Dataset",     
        output_dir="Data\segmentationData", 
        train_ratio=0.8                      # 80% training, 20% testing
    )