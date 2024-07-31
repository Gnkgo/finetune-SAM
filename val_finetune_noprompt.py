# import necessary libraries
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from pathlib import Path
import json
from argparse import Namespace

from models.sam import SamPredictor, sam_model_registry
from models.sam.utils.transforms import ResizeLongestSide
from models.sam_LoRa import LoRA_Sam
from utils.dataset import Public_dataset
from utils.losses import DiceLoss
from utils.dsc import dice_coeff
from utils.utils import vis_image
import cfg

def calculate_metrics(pred, target):
    TP = ((pred == 1) & (target == 1)).sum().item()
    TN = ((pred == 0) & (target == 0)).sum().item()
    FP = ((pred == 1) & (target == 0)).sum().item()
    FN = ((pred == 0) & (target == 1)).sum().item()
    return TP, TN, FP, FN

def main(args, test_image_list):
    test_dataset = Public_dataset(args, args.img_folder, args.mask_folder, test_img_list, phase='val', targets=['combine_all'], if_prompt=False)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    cls_num = 2

    if args.finetune_type == 'adapter' or args.finetune_type == 'vanilla':
        sam_fine_tune = sam_model_registry[args.arch](args, checkpoint=os.path.join(args.dir_checkpoint, 'checkpoint_best.pth'), num_classes=cls_num)
    elif args.finetune_type == 'lora':
        sam = sam_model_registry[args.arch](args, checkpoint=os.path.join(args.sam_ckpt), num_classes=cls_num)
        sam_fine_tune = LoRA_Sam(args, sam, r=4).to('cuda').sam
        sam_fine_tune.load_state_dict(torch.load(args.dir_checkpoint + '/checkpoint_best.pth'), strict=False)
        
    sam_fine_tune = sam_fine_tune.to('cuda').eval()
    class_iou = torch.zeros(args.num_cls, dtype=torch.float)
    cls_dsc = torch.zeros(args.num_cls, dtype=torch.float)
    eps = 1e-9
    img_name_list = []
    pred_msk = []
    test_img = []
    test_gt = []

    metrics = {"Dice": [], "FN": [], "FP": [], "IoU": [], "TN": [], "TP": [], "n_pred": [], "n_ref": []}

    for i, data in enumerate(tqdm(testloader)):
        imgs = torchvision.transforms.Resize((1024, 1024))(data['image']).to('cuda')
        msks = torchvision.transforms.Resize((args.out_size, args.out_size))(data['mask']).to('cuda')
        img_name_list.append(data['img_name'][0])

        with torch.no_grad():
            img_emb = sam_fine_tune.image_encoder(imgs)
            sparse_emb, dense_emb = sam_fine_tune.prompt_encoder(points=None, boxes=None, masks=None)
            pred_fine, _ = sam_fine_tune.mask_decoder(image_embeddings=img_emb, image_pe=sam_fine_tune.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_emb, dense_prompt_embeddings=dense_emb, multimask_output=True)
           
        pred_fine = pred_fine.argmax(dim=1)

        pred_msk.append(pred_fine.cpu())
        test_img.append(imgs.cpu())
        test_gt.append(msks.cpu())
        yhat = pred_fine.cpu().long().flatten()
        y = msks.cpu().flatten()

        for j in range(args.num_cls):
            y_bi = y == j
            yhat_bi = yhat == j
            I = (y_bi * yhat_bi).sum().item()
            U = torch.logical_or(y_bi, yhat_bi).sum().item()
            class_iou[j] += I / (U + eps)

            mask_pred_cls = (pred_fine.cpu() == j).float()
            mask_gt_cls = (msks.cpu() == j).float()
            cls_dsc[j] += dice_coeff(mask_pred_cls, mask_gt_cls).item()
            
            TP, TN, FP, FN = calculate_metrics(mask_pred_cls, mask_gt_cls)
            metrics["TP"].append(TP)
            metrics["TN"].append(TN)
            metrics["FP"].append(FP)
            metrics["FN"].append(FN)
            metrics["n_pred"].append(mask_pred_cls.sum().item())
            metrics["n_ref"].append(mask_gt_cls.sum().item())

        print(i)

    class_iou /= (i + 1)
    cls_dsc /= (i + 1)

    save_folder = os.path.join('/cluster/home/jbrodbec/finetune-SAM/test_results', args.dir_checkpoint)
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(save_folder, 'test_masks.npy'), np.concatenate(pred_msk, axis=0))
    np.save(os.path.join(save_folder, 'test_name.npy'), np.concatenate(np.expand_dims(img_name_list, 0), axis=0))

    print(dataset_name)      
    print('class dsc:', cls_dsc)      
    print('class iou:', class_iou)

    overall_metrics = {
        "foreground_mean": {
            "Dice": cls_dsc.mean().item(),
            "FN": np.mean(metrics["FN"]),
            "FP": np.mean(metrics["FP"]),
            "IoU": class_iou.mean().item(),
            "TN": np.mean(metrics["TN"]),
            "TP": np.mean(metrics["TP"]),
            "n_pred": np.mean(metrics["n_pred"]),
            "n_ref": np.mean(metrics["n_ref"])
        }
    }

    with open(os.path.join(save_folder, 'overall_metrics.json'), 'w') as f:
        json.dump(overall_metrics, f, indent=4)
    print(overall_metrics)

if __name__ == "__main__":
    args = cfg.parse_args()

    if 1:
        args_path = f"{args.dir_checkpoint}/args.json"

        with open(args_path, 'r') as f:
            args_dict = json.load(f)
        
        args = Namespace(**args_dict)
        
    dataset_name = args.dataset_name
    print('train dataset: {}'.format(dataset_name)) 
    test_img_list = args.img_folder + dataset_name + '/val.csv'
    main(args, test_img_list)
