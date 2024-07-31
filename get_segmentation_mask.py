from argparse import Namespace
from pathlib import Path
from PIL import Image
import json
import numpy as np
import os
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from models.sam import SamPredictor, sam_model_registry
from utils.utils import vis_image, inverse_normalize, torch_percentile


#print(f"Using device: {device}")


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_config(args_path):
    with open(args_path, 'r') as f:
        args_dict = json.load(f)
    return Namespace(**args_dict)

# Function to evaluate a single image slice
def evaluate_1_slice(image_path, model, device):
    """
    Evaluates a single image slice using the provided model.

    Parameters:
    - image_path: Path to the image slice file.
    - model: The model used for evaluation.

    Returns:
    - ori_img: The original image after normalization.
    - pred: The prediction from the model.
    - Pil_img: The PIL image of the original slice.
    """
    # Load the image
    img = Image.open(image_path).convert('RGB')
    Pil_img = img.copy()

    # Resize the image to 1024x1024
    img = transforms.Resize((1024, 1024))(img)

    # Transform the image to a tensor and normalize
    transform_img = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = transform_img(img)
    imgs = torch.unsqueeze(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img), 0).to(device)

    # Perform model inference without gradient calculation
    with torch.no_grad():
        # Get image embeddings from the image encoder
        img_emb = model.image_encoder(imgs)

        # Get sparse and dense embeddings from the prompt encoder
        sparse_emb, dense_emb = model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )

        # Get the prediction from the mask decoder
        pred, _ = model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
        )

        # Get the most likely prediction
        pred = pred.argmax(dim=1)

    # Get the original image after normalization
    ori_img = inverse_normalize(imgs.cpu()[0])

    return ori_img, pred, Pil_img

def run_folder(input_dir, output_dir, args_path, checkpoint_dir, device):
    args = load_config(args_path)
    sam_fine_tune = sam_model_registry[args.arch](args, checkpoint=os.path.join(checkpoint_dir, 'finetune-sam.pth'), num_classes=args.num_cls)
    sam_fine_tune = sam_fine_tune.to(device).eval()
    for image_file in Path(input_dir).glob("*.png"):  # You can change the extension to match your image format
        ori_img, pred, Pil_img = evaluate_1_slice(image_file, sam_fine_tune, device)
        mask_pred = ((pred).cpu()).float()
        pil_mask = Image.fromarray(np.array(mask_pred[0], dtype=np.uint8), 'L').resize(Pil_img.size, Image.NEAREST)

        mask_img_filename = os.path.join(output_dir, f'{image_file.stem}.png')
        pil_mask.save(mask_img_filename)

        #print(f"Predicted mask saved to {mask_img_filename}")

def run_nnunetv2_predict(args):
    # Construct the command with the specific parameters
    command = [
        "-d", "Dataset011_nasare",
        "-i", args.resized_images,
        "-o", args.segmentation_mask_scar,
        "-f", "3",
        "-tr", "nnUNetTrainer",
        "-c", "2d",
        "-p", "nnUNetPlans",
        "-device", args.device
    ]

    #predict_entry_point(command)



# Example usage
#C:\Users\Joanna Brodbeck\Documents\GitHub\mastectomy\nasare\models\segmentation_model\nnUNet_results\Dataset011_nasare\nnUNetTrainer__nnUNetPlans__2d\dataset.json
#C:/Users/Joanna Brodbeck/Documents/GitHub/mastectomy/nasare/model/segmentation_model/nnUNet_results/Dataset011_nasare/nnUNetTrainer__nnUNetPlans__2d/dataset.json

def run_nnunetv2_apply_postprocessing(args):
    # Construct the command for nnUNetv2_apply_postprocessing
    nnUNet_results = os.getenv("nnUNet_results", None)
    if not nnUNet_results:
        raise EnvironmentError("Environment variable 'nnUNet_results' is not set.")

    postprocess_command = [
        "-i",  args.segmentation_mask_scar,
        "-o",  args.segmentation_mask_scar,
        "-pp_pkl_file", args.pp_pkl_file,
        "-np", "8",
        "-plans_json", args.plans_json
    ]

    #entry_point_apply_postprocessing(postprocess_command)

def rename_files(input_dir, flag):
    for image_file in Path(input_dir).glob("*.png"):
        if flag:
            new_name = image_file.stem.split("_0000")[0] + ".png"
        else:
            new_name = image_file.stem + "_0000.png"
        os.rename(image_file, os.path.join(input_dir, new_name))


def main():

    inpainting_model = "finetune_sam"

    if (inpainting_model == "finetune_sam"):
        checkpoint_dir = "./checkpoints2"
        args_path = f"{checkpoint_dir}/args.json"
        input_dir = "./images"
        output_dir = "./results"
        device = torch.device("cpu")
        #os.makedirs(output_dir, exist_ok=True)
        run_folder(input_dir, output_dir, args_path, checkpoint_dir, device)

    # elif (args.inpainting_model == "nnunet"):
    #     rename_files(args.resized_images, 0)
    #     run_nnunetv2_predict(args)
    #     run_nnunetv2_apply_postprocessing(args)
    #     rename_files(args.resized_images, 1)
    #     rename_files(args.segmentation_mask_scar, 1)
    #     #torch.set_num_threads(1)

    else:
        print("Invalid inpainting model")
        return


    #print("checkpoint loaded")
    print(f"Segmentation mask for scars created")


if __name__ == "__main__":
    main()
