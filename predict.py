import os
import argparse
import torch
import timm
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from utils.helpers import read_config, get_device, generate_dirs, seed_everything, set_pytorch_optimizations
from utils.logger import setup_logger
from datasets.classification_dataset import ClassificationDataset
from datasets.segmentation_dataset import SegmentationDataset
from utils.transforms import val_transforms_classification, val_transform_segmentation
from utils.visualize import display_classification_batch, display_segmentation_batch
from tqdm.auto import tqdm

# Set PyTorch performance optimizations
set_pytorch_optimizations()

def setup_environment(config_path, log_filename):
    configs = read_config(config_path)
    device = get_device()
    generate_dirs(configs.get("inference", {}))
    seed_everything(configs["seed"])
    logger = setup_logger(os.path.join(configs["inference"]["output_dir"], log_filename))
    return configs, device, logger


def run_classification_inference(configs, device, logger, img_dir, checkpoint_path):
    model_cfg = configs["model"]
    data_cfg = configs["data"]
    inference_cfg = configs.get("inference", {})

    batch_size = inference_cfg["batch_size"]
    num_workers = inference_cfg["num_workers"]

    logger.info("Classification evaluation....")
    model = timm.create_model(model_cfg["name"], pretrained=model_cfg["pretrained"], num_classes=model_cfg["num_classes"])
    model.to(device)
    logger.info(f"Loading classification model: {model_cfg['name']} with {model_cfg['num_classes']} classes.")
    
    if not checkpoint_path:
        raise ValueError("Checkpoint path is required for classification inference.")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    logger.info(f"Model checkpoint loaded from {checkpoint_path}")

    test_data = ClassificationDataset(img_dir, transform=val_transforms_classification, is_inference=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    os.makedirs(inference_cfg["output_dir"], exist_ok=True)
    all_predictions = []

    logger.info(f"Starting inference on {len(test_data)} images...")

    with torch.no_grad():
        for idx, (images, filenames) in enumerate(tqdm(test_dataloader, desc="Inferencing the dataset", leave=False)):
            
            images = images.to(device)
            preds = model(images)
            pred_classes = torch.argmax(preds, dim=1)

            display_classification_batch(images, pred_classes, idx, inference_cfg, n=len(images))
            for fname, pred in zip(filenames, preds.cpu().tolist()):
                all_predictions.append((fname, pred))
                logger.info(f"Predicted: {fname} => class {torch.argmax(torch.tensor(pred)).item()}")
       
        logger.info("Classification inference completed.")
    
    return all_predictions


def run_segmentation_inference(configs, device, logger, img_dir, checkpoint_path):
    model_cfg = configs["model"]
    data_cfg = configs["data"]
    inference_cfg = configs.get("inference", {})
    
    batch_size = inference_cfg["batch_size"]
    num_workers = inference_cfg["num_workers"]

    logger.info("Segmentation evaluation....")
    model = smp.Unet(
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["classes"]
    )
    model.to(device)

    logger.info(f"Loading segmentation model: Unet with {model_cfg['encoder_name']} encoder and {model_cfg['classes']} classes.")
    
    if not checkpoint_path:
        raise ValueError("Checkpoint path is required for segmentation inference.")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    logger.info(f"Model checkpoint loaded from {checkpoint_path}")

    test_data = SegmentationDataset(img_dir, val_transform_segmentation, is_inference=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    os.makedirs(inference_cfg["output_dir"], exist_ok=True)
    all_predictions = []

    logger.info(f"Starting inference on {len(test_data)} images...")
    
    for batch_idx, (images, filenames) in enumerate(tqdm(test_dataloader, desc="Inferencing the dataset")):
        
        images = images.to(device)
        preds = model(images)
        preds = torch.argmax(preds, dim=1)
        
        display_segmentation_batch(images, preds, batch_idx, inference_cfg, n=len(images))
        
        logger.info(f"Processed and saved segmentation output for batch including {', '.join(filenames)}")
    
    logger.info("Segmentation inference completed. Outputs saved to the specified output directory.")
    
    return all_predictions

def main():
    parser = argparse.ArgumentParser(description="pytorch based framework for classification and segmentation tasks")
    parser.add_argument("--config_path", type=str, required=True, help="Path of the config file")
    parser.add_argument("--img_dir", type=str, help="Path to test data")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint model - state dict")
    args = parser.parse_args()

    configs, device, logger = setup_environment(args.config_path, 'log_inference.txt')
    task_type = configs["task_type"]
    if task_type == "classification":
        run_classification_inference(configs, device, logger, args.img_dir, args.checkpoint_path)
    elif task_type == "segmentation":
        run_segmentation_inference(configs, device, logger, args.img_dir, args.checkpoint_path)
    else:
        logger.error(f"Unknown task_type: {task_type}")
        raise ValueError(f"Unknown task_type: {task_type}")

if __name__ == "__main__":
    main()

