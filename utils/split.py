import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

def prepare_classification_split(data_dir: str, csv_name: str = "data.csv", 
                                  train_ratio: float = 0.8, seed: int = 42):
    """
    Prepares train/val splits for image classification.
    
    Args:
        data_dir (str): Root directory containing images in subdirectories and a CSV file.
        csv_name (str): Name of the CSV file with columns [Image-Name, Label].
        train_ratio (float): Ratio of data to include in the training set.
        seed (int): Random seed for reproducibility.
    """

    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / csv_name)

    # Add ".png" and find full image path
    df['Image-File'] = df['Image-Name'].astype(str) + '.png'

    # Search recursively in subdirs for images
    all_image_paths = list(data_dir.rglob("*.png"))
    image_map = {img_path.stem: img_path for img_path in all_image_paths}

    # Map to actual full path
    df['Full-Path'] = df['Image-Name'].map(image_map)

    # Drop rows without actual file match
    df = df.dropna(subset=['Full-Path'])

    # Split
    train_df, val_df = train_test_split(df, train_size=train_ratio, stratify=df['Label'], random_state=seed)

    # Prepare output dirs
    for split_name, split_df in zip(['train', 'val'], [train_df, val_df]):
        split_dir = data_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        # Copy images and prepare new CSV
        new_rows = []
        for _, row in split_df.iterrows():
            src_path = row['Full-Path']
            dst_path = split_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            new_rows.append({'Image-Name': src_path.name, 'Label': row['Label']})
        
        pd.DataFrame(new_rows).to_csv(split_dir / f"{split_name}.csv", index=False)

    print("✅ Train/Val split completed!")

prepare_classification_split(r"C:\Users\e87299\Desktop\Training\Week2-Pytorch\Framework-pytorch\Classification-Data", csv_name="label.csv")
