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
from datasets.classification_dataset import ClassificationInferenceDataset
from datasets.segmentation_dataset import SegmentationData
from utils.visualize import display_classification_batch, display_classification_prediction
from utils.transforms import val_transforms_classification, val_transform_segmentation
from utils.helpers import read_config, get_device, generate_dirs

import matplotlib.pyplot as plt


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = True

def setup_logger(log_file):
    logging.basicConfig(
        filename = log_file,
        encoding = "utf-8",
        level=logging.DEBUG,
        format = '%(levelname)s - %(message)s'
    )

    return logging.getLogger()


def main():
    parser = argparse.ArgumentParser(description="pytorch based framework for classifcation and segmentation tasks")
    parser.add_argument("--config_path", type=str, required=True, help="Path of the config file")
    parser.add_argument("--img_dir", type = str, help = "path to test data")
    parser.add_argument("--checkpoint_path", type = str, help = "Path to the checkpoint model - state dict")
    args = parser.parse_args()

    configs = read_config(args.config_path)
    device = get_device()

    logger = setup_logger(os.path.join(configs["output_dir"], 'log.txt'))
    logger.info("Starting main processing")
    logger.info(f"Using device: {device}")

    generate_dirs(configs)

    if(configs['task_type'] == '0'):

        logger.info("Classification evaluation....")
    
        model = timm.create_model('resnet50d', pretrained=False, num_classes=configs['num_classes'])
        model.to(device)

        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        # model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['model'].state_dict())  # Using .state_dict() from saved model
        model.eval()

        test_data = ClassificationInferenceDataset(args.img_dir, transform=val_transforms_classification)
        test_dataloader = DataLoader(test_data, batch_size=configs['batch_size'], num_workers=configs['num_workers'], shuffle=False)

        os.makedirs(configs["output_dir"], exist_ok=True)
        all_predictions = []

        with torch.no_grad():
            for batch_idx, (images, filenames) in enumerate(tqdm(test_dataloader, desc="Inferencing the dataset", leave=False)):
                images = images.to(device)

                preds = model(images)
                pred_probs = torch.softmax(preds, dim=1)
                pred_classes = torch.argmax(pred_probs, dim=1)

                display_classification_batch(images, preds, configs)

                for fname, pred in zip(filenames, preds.cpu().tolist()):
                    all_predictions.append((fname, pred))
                        
            for fname, pred in all_predictions:
                print(f"{fname} => class {pred}")

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

        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'].state_dict())  # Using .state_dict() from saved model
        # model.load_state_dict(checkpoint)
        model.eval()

        img = Image.open(args.img_path)
        input_tensor = val_transform_segmentation(image=np.array(img))['image']  
        input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dim and send to device

        with torch.no_grad():
            output = model(input_tensor)
            print(output.shape)
            prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            print(prediction.shape)

        plt.imshow(prediction, cmap='jet')
        plt.title("Predicted Segmentation")
        plt.axis('off')
        plt.show()
                    
        logger.info("Segmentation output saved as 'output.png'")


if __name__ == "__main__":
    main()