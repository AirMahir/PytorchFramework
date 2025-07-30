import os
import argparse
import torch
import numpy as np
import logging
import timm
import segmentation_models_pytorch as smp

from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from datasets.classification_dataset import ClassificationDataset
from datasets.segmentation_dataset import SegmentationDataset
from utils.visualize import display_classification_batch, display_segmentation_batch
from utils.transforms import val_transforms_classification, val_transform_segmentation
from utils.helpers import read_config, get_device, generate_dirs, seed_everything

import matplotlib.pyplot as plt


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = True

def setup_logger(log_file):
    logging.basicConfig(
        filename = log_file,
        encoding = "utf-8",
        level=logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Adding a console handler to see logs in terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    # Get the root logger and add the console handler
    root_logger = logging.getLogger()
    if not root_logger.handlers: # Avoid adding duplicate handlers if setup_logger is called multiple times
        root_logger.addHandler(console_handler)
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="pytorch based framework for classifcation and segmentation tasks")
    parser.add_argument("--config_path", type=str, required=True, help="Path of the config file")
    parser.add_argument("--img_dir", type = str, help = "path to test data")
    parser.add_argument("--checkpoint_path", type = str, help = "Path to the checkpoint model - state dict")
    args = parser.parse_args()

    configs = read_config(args.config_path)
    device = get_device()

    generate_dirs(configs)
    seed_everything(configs["seed"])

    logger = setup_logger(os.path.join(configs["output_dir"], 'log_inference.txt'))
    logger.info("Starting main processing")
    logger.info(f"Using device: {device}")

    if(configs['task_type'] == '0'):

        logger.info("Classification evaluation....")
    
        model = timm.create_model('resnet50d', pretrained=False, num_classes=configs['num_classes'])
        model.to(device)
        logger.info(f"Loading classification model: resnet50d with {configs['num_classes']} classes.")
        if not args.checkpoint_path:
            raise ValueError("Checkpoint path is required for classification inference.")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        # model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['model'])  # Using .state_dict() from saved model
        model.eval()
        logger.info(f"Model checkpoint loaded from {args.checkpoint_path}")

        test_data = ClassificationDataset(args.img_dir, transform=val_transforms_classification, is_inference=True)
        test_dataloader = DataLoader(test_data, batch_size=configs['batch_size'], num_workers=configs['num_workers'], shuffle=False)

        os.makedirs(configs["output_dir"], exist_ok=True)
        all_predictions = []

        logger.info(f"Starting inference on {len(test_data)} images...")

        with torch.no_grad():
            for batch_idx, (images, filenames) in enumerate(tqdm(test_dataloader, desc="Inferencing the dataset", leave=False)):
                images = images.to(device)

                preds = model(images)
                pred_classes = torch.argmax(preds, dim=1)
                
                display_classification_batch(images, pred_classes, configs)

                for fname, pred in zip(filenames, preds.cpu().tolist()):
                    all_predictions.append((fname, pred))
                    logger.info(f"Predicted: {fname} => class {torch.argmax(torch.tensor(pred)).item()}")
                        
            logger.info("Classification inference completed.")

            return all_predictions

    else:
        logger.info("Segmentation evaluation....")

        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=3
        )
        model.to(device)
        logger.info(f"Loading segmentation model: Unet with resnet34 encoder and {3} classes.")

        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])  
        model.eval()
        logger.info(f"Model checkpoint loaded from {args.checkpoint_path}")

        test_data = SegmentationDataset(args.img_dir, val_transform_segmentation, is_inference=True)
        test_dataloader = DataLoader(test_data, batch_size = configs['batch_size'], num_workers=configs['num_workers'], shuffle=True)

        os.makedirs(configs["output_dir"], exist_ok=True)

        all_predictions = []

        logger.info(f"Starting inference on {len(test_data)} images...")

        for batch_idx, (images, filenames) in enumerate(tqdm(test_dataloader, desc="Inferencing the dataset")):

            print(f"Batch size: {len(images)}")
            images = images.to(device)
            preds = model(images)
            preds = torch.argmax(preds, dim=1)

            display_segmentation_batch(images, preds, batch_idx, configs, n=len(images))
            logger.info(f"Processed and saved segmentation output for batch including {', '.join(filenames)}")
                    
        logger.info("Segmentation inference completed. Outputs saved to the specified output directory.")


if __name__ == "__main__":
    main()