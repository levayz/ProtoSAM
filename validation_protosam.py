"""
Validation script
"""
import math
import os
import pandas as pd
import csv
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import time
import matplotlib.pyplot as plt
from models.ProtoSAM import ProtoSAM,  ALPNetWrapper, SamWrapperWrapper, InputFactory, ModelWrapper, TYPE_ALPNET, TYPE_SAM
from models.ProtoMedSAM import ProtoMedSAM
from models.grid_proto_fewshot import FewShotSeg
from models.segment_anything.utils.transforms import ResizeLongestSide
from models.SamWrapper import SamWrapper
# from dataloaders.PolypDataset import get_polyp_dataset, get_vps_easy_unseen_dataset, get_vps_hard_unseen_dataset, PolypDataset, KVASIR, CVC300, COLON_DB, ETIS_DB, CLINIC_DB
from dataloaders.PolypDataset import get_polyp_dataset, PolypDataset
from dataloaders.PolypTransforms import get_polyp_transform
from dataloaders.SimpleDataset import SimpleDataset
from dataloaders.ManualAnnoDatasetv2 import get_nii_dataset
from dataloaders.common import ValidationDataset
from config_ssl_upload import ex

import tqdm
from tqdm.auto import tqdm
import cv2
from collections import defaultdict

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

# Supported Datasets
CHAOS = "chaos"
SABS = "sabs"
POLYPS = "polyps"

ALP_DS = [CHAOS, SABS]

ROT_DEG = 0

def get_bounding_box(segmentation_map):
    """Generate bounding box from a segmentation map. one bounding box to include the extreme points of the segmentation map."""
    if isinstance(segmentation_map, torch.Tensor):
        segmentation_map = segmentation_map.cpu().numpy()
    
    bbox = cv2.boundingRect(segmentation_map.astype(np.uint8))
    # plot bounding boxes for each contours
    # plt.figure()
    # x, y, w, h = bbox
    # plt.imshow(segmentation_map)
    # plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='r', linewidth=2))
    # plt.savefig("debug/bounding_boxes.png") 

    return bbox

def calc_iou(boxA, boxB):
    """
    boxA: [x, y, w, h]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def eval_detection(pred_list):
    """
    pred_list: list of dictionaries with keys 'pred_bbox', 'gt_bbox' and score (prediction confidence score).
    compute AP50, AP75, AP50:95:10
    """
    iou_thresholds = np.round(np.arange(0.5, 1.0, 0.05), 2)
    ap_dict = {iou: [] for iou in iou_thresholds}
    for iou_threshold in iou_thresholds:
        tp, fp = 0, 0
        
        for pred in pred_list:
            pred_bbox = pred['pred_bbox']
            gt_bbox = pred['gt_bbox']
            
            iou = calc_iou(pred_bbox, gt_bbox)
            
            if iou >= iou_threshold:
                tp += 1
            else:
                fp += 1

        precision = tp / (tp + fp)
        recall = tp / len(pred_list) 
        f1 = 2 * (precision * recall) / (precision + recall)        

        ap_dict[iou_threshold] = {
            'iou_threshold': iou_threshold,
            'tp': tp,
            'fp': fp,
            'n_gt': len(pred_list),
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # Convert results to a DataFrame and save to CSV
    results = []
    for iou_threshold in iou_thresholds:
        results.append(ap_dict[iou_threshold])
    
    df = pd.DataFrame(results)
    return df


def plot_pred_gt_support(query_image, pred, gt, support_images, support_masks, score=None, save_path="debug/pred_vs_gt.png"):
    """
    pred: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
    gt: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
    support: 4d tensor of shape (N, C, H, W) where 1 represents foreground and 0 represents background
    """
    if support_images:
        if isinstance(support_images, list):
            support_images = torch.cat(support_images, dim=0).clone().detach()
        if isinstance(support_masks, list):
            support_masks = torch.cat(support_masks, dim=0).clone().detach()
        if len(query_image.shape) == 3:
            query_image = query_image.permute(1, 2, 0).clone().detach()
        if len(support_images.shape) == 4:
            support_images = support_images.clone().detach().permute(0, 2, 3, 1)
        n_support_rows = math.ceil(support_images.shape[0] / 2)
    else:
        n_support_rows = 1
    fig, ax = plt.subplots(n_support_rows + 1, 2)
    query_image = (query_image - query_image.min()) / (query_image.max() - query_image.min())
    ax[0, 0].imshow(query_image.cpu().detach())
    ax[0, 0].imshow(pred, alpha=0.5)
    ax[0, 0].set_title("pred")
    ax[0, 1].imshow(query_image.cpu().detach())
    ax[0, 1].imshow(gt, alpha=0.5)
    ax[0, 1].set_title("gt")
    if support_images is not None:
        for i in range(1, n_support_rows + 1):
            support_images[(i - 1) * 2] = (support_images[(i - 1) * 2] - support_images[(i - 1) * 2].min()) / (support_images[(i - 1) * 2].max() - support_images[(i - 1) * 2].min())
            ax[i, 0].imshow(support_images[(i - 1) * 2].cpu().detach())
            ax[i, 0].imshow(support_masks[(i - 1) * 2].cpu(), alpha=0.5)
            ax[i, 0].set_title(f"support")
            if (i - 1) * 2 + 1 < support_images.shape[0]:
                support_images[(i - 1) * 2 + 1] = (support_images[(i - 1) * 2 + 1] - support_images[(i - 1) * 2 + 1].min()) / (support_images[(i - 1) * 2 + 1].max() - support_images[(i - 1) * 2 + 1].min())
                ax[i, 1].imshow(support_images[(i - 1) * 2 + 1].cpu().detach())
                ax[i, 1].imshow(support_masks[(i - 1) * 2 + 1].cpu(), alpha=0.5)
                ax[i, 1].set_title(f"support")
    if score is not None:
        # plt.title(f"score: {score}") 
        fig.suptitle(f"sam score: {score}")
    fig.savefig(save_path)
    plt.close(fig)


def get_dice_iou_precision_recall(pred: torch.Tensor, gt: torch.Tensor):
    """
    pred: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
    gt: 2d tensor of shape (H, W) where 1 represents foreground and 0 represents background
    """
    if gt.sum() == 0:
        print("gt is all background")
        return {"dice": 0, "precision": 0, "recall": 0}

    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def get_alpnet_model(_config) -> ModelWrapper:
    alpnet = FewShotSeg(
       _config["input_size"][0],
       _config["reload_model_path"],
       _config["model"]
    )
    alpnet.cuda()
    alpnet_wrapper = ALPNetWrapper(alpnet)
    
    return alpnet_wrapper

def get_sam_model(_config) -> ModelWrapper:
    sam_args = {
        "model_type": "vit_h",
        "sam_checkpoint": "pretrained_model/sam_vit_h.pth"
    }
    sam = SamWrapper(sam_args=sam_args).cuda()
    sam_wrapper = SamWrapperWrapper(sam)
    return sam_wrapper  

def get_model(_config) -> ProtoSAM:
    # Initial Segmentation Model
    if _config["base_model"] == TYPE_ALPNET:
        base_model = get_alpnet_model(_config)
    else:
        raise NotImplementedError(f"base model {_config['base_model']} not implemented")
    
    # ProtoSAM model
    if _config["protosam_sam_ver"] in  ("sam_h", "sam_b"):
        sam_h_checkpoint = "pretrained_model/sam_vit_h.pth"
        sam_b_checkpoint = "pretrained_model/sam_vit_b.pth"
        sam_checkpoint = sam_h_checkpoint if _config["protosam_sam_ver"] == "sam_h" else sam_b_checkpoint
        model = ProtoSAM(image_size = (1024, 1024),
                    coarse_segmentation_model=base_model,
                    use_bbox=_config["use_bbox"],
                    use_points=_config["use_points"],
                    use_mask=_config["use_mask"],
                    debug=_config["debug"],
                    num_points_for_sam=1,
                    use_cca=_config["do_cca"],
                    point_mode=_config["point_mode"],
                    use_sam_trans=True, 
                    coarse_pred_only=_config["coarse_pred_only"],
                    sam_pretrained_path=sam_checkpoint,
                    use_neg_points=_config["use_neg_points"],) 
    elif _config["protosam_sam_ver"] == "medsam":
        model = ProtoMedSAM(image_size = (1024, 1024),
                            coarse_segmentation_model=base_model,
                            debug=_config["debug"],
                            use_cca=_config["do_cca"],
        )
    else:
        raise NotImplementedError(f"protosam_sam_ver {_config['protosam_sam_ver']} not implemented")
    
    return model


def get_support_set_polyps(_config, dataset:PolypDataset):
    n_support = _config["n_support"]
    (support_images, support_labels, case) = dataset.get_support(n_support=n_support)
    
    return support_images, support_labels, case


def get_support_set_alpds(config, dataset:ValidationDataset):
    support_set = dataset.get_support_set(config)
    support_fg_masks = support_set["support_labels"]
    support_images = support_set["support_images"]
    support_scan_id = support_set["support_scan_id"]
    return support_images, support_fg_masks, support_scan_id


def get_support_set(_config, dataset):
    if _config["dataset"].lower() == POLYPS:
        support_images, support_fg_masks, case = get_support_set_polyps(_config, dataset)
    elif any(item in _config["dataset"].lower() for item in ALP_DS):
        support_images, support_fg_masks, support_scan_id = get_support_set_alpds(_config, dataset)
    else:
        raise NotImplementedError(f"dataset {_config['dataset']} not implemented")
    return support_images, support_fg_masks, support_scan_id


def update_support_set_by_scan_part(support_images, support_labels, qpart):
    qpart_support_images = [support_images[qpart]]
    qpart_support_labels = [support_labels[qpart]]
    
    return qpart_support_images, qpart_support_labels


def manage_support_sets(sample_batched, all_support_images, all_support_fg_mask, support_images, support_fg_mask, qpart=None):
    if sample_batched['part_assign'][0] != qpart:
        qpart = sample_batched['part_assign'][0]
        support_images, support_fg_mask = update_support_set_by_scan_part(all_support_images, all_support_fg_mask, qpart)
            
    return support_images, support_fg_mask, qpart


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        print(f"####### created dir:{_run.observers[0].dir} #######")
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')
    print(f"config do_cca: {_config['do_cca']}, use_bbox: {_config['use_bbox']}")
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'###### Reload model {_config["reload_model_path"]} ######')
    model = get_model(_config)
    model = model.to(torch.device("cuda"))
    model.eval()
    
    sam_trans = ResizeLongestSide(1024)
    if _config["dataset"].lower() == POLYPS:
        tr_dataset, te_dataset = get_polyp_dataset(sam_trans=sam_trans, image_size=(1024, 1024))
    elif CHAOS in _config["dataset"].lower() or SABS in _config["dataset"].lower():
        tr_dataset, te_dataset = get_nii_dataset(_config, _config["input_size"][0]) 
    else:
        raise NotImplementedError(
            f"dataset {_config['dataset']} not implemented")

    # dataloaders
    testloader = DataLoader(
        te_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False
    )

    _log.info('###### Starting validation ######')
    model.eval()

    mean_dice = []
    mean_prec = []
    mean_rec = []
    mean_iou = []
    
    mean_dice_cases = {}
    mean_iou_cases = {} 
    bboxes_w_scores = []
    
    curr_case = None
    supp_fts = None
    qpart = None
    support_images = support_fg_mask = None
    all_support_images, all_support_fg_mask, support_scan_id = None, None, None
    MAX_SUPPORT_IMAGES = 1
    is_alp_ds = any(item in _config["dataset"].lower() for item in ALP_DS)
    is_polyp_ds  = _config["dataset"].lower() == POLYPS
    
    if is_alp_ds:
        all_support_images, all_support_fg_mask, support_scan_id = get_support_set(_config, te_dataset)
    elif is_polyp_ds:
        support_images, support_fg_mask, case = get_support_set_polyps(_config, tr_dataset)
        
    with tqdm(testloader) as pbar: 
        for idx, sample_batched in enumerate(tqdm(testloader)):
            case = sample_batched['case'][0]
            if is_alp_ds: 
                support_images, support_fg_mask, qpart = manage_support_sets(
                                                            sample_batched,
                                                            all_support_images,
                                                            all_support_fg_mask,
                                                            support_images,
                                                            support_fg_mask,
                                                            qpart,
                )
            
            if is_alp_ds and sample_batched["scan_id"][0] in support_scan_id:
                continue
             
            query_images = sample_batched['image'].cuda()
            query_labels = torch.cat([sample_batched['label']], dim=0)
            if not 1 in query_labels and _config["skip_no_organ_slices"]:
                continue
            
            n_try = 1
            with torch.no_grad():
                coarse_model_input = InputFactory.create_input(
                                        input_type=_config["base_model"],
                                        query_image=query_images,
                                        support_images=support_images,
                                        support_labels=support_fg_mask,
                                        isval=True,
                                        val_wsize=_config["val_wsize"],
                                        original_sz=query_images.shape[-2:],
                                        img_sz=query_images.shape[-2:],
                                        gts=query_labels,
                )
                coarse_model_input.to(torch.device("cuda"))
                    
                query_pred, scores = model(
                        query_images, coarse_model_input, degrees_rotate=0)
            query_pred = query_pred.cpu().detach()
                
            if _config["debug"]:
                if is_alp_ds:
                    save_path = f'debug/preds/{case}_{sample_batched["z_id"].item()}_{idx}_{n_try}.png'
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                elif is_polyp_ds:
                    save_path = f'debug/preds/{case}_{idx}_{n_try}.png'
                plot_pred_gt_support(query_images[0,0].cpu(), query_pred.cpu(), query_labels[0].cpu(
                ), support_images, support_fg_mask, save_path=save_path, score=scores[0])

            metrics = get_dice_iou_precision_recall(
                query_pred, query_labels[0].to(query_pred.device))
            mean_dice.append(metrics["dice"])
            mean_prec.append(metrics["precision"])
            mean_rec.append(metrics["recall"])
            mean_iou.append(metrics["iou"])

            bboxes_w_scores.append({"pred_bbox": get_bounding_box(query_pred.cpu()),
                                    "gt_bbox": get_bounding_box(query_labels[0].cpu()),
                                    "score": np.mean(scores)})
            
            if case not in mean_dice_cases:
                mean_dice_cases[case] = []
                mean_iou_cases[case] = []
            mean_dice_cases[case].append(metrics["dice"])
            mean_iou_cases[case].append(metrics["iou"])

            if metrics["dice"] < 0.6 and _config["debug"]:
                path = f'{_run.observers[0].dir}/bad_preds/case_{case}_idx_{idx}_dice_{metrics["dice"]:.4f}.png'
                if _config["debug"]:
                    path = f'debug/bad_preds/case_{case}_idx_{idx}_dice_{metrics["dice"]:.4f}.png'
                os.makedirs(os.path.dirname(path), exist_ok=True)
                print(f"saving bad prediction to {path}")
                plot_pred_gt_support(query_images[0,0].cpu(), query_pred.cpu(), query_labels[0].cpu(
                    ), support_images, support_fg_mask, save_path=path, score=scores[0])
                
            pbar.set_postfix_str({"mdice": f"{np.mean(mean_dice):.4f}", "miou": f"{np.mean(mean_iou):.4f}, n_try: {n_try}"})
                

    for k in mean_dice_cases.keys():
        _run.log_scalar(f'mar_val_batches_meanDice_{k}', np.mean(mean_dice_cases[k]))
        _run.log_scalar(f'mar_val_batches_meanIOU_{k}', np.mean(mean_iou_cases[k]))
        _log.info(f'mar_val batches meanDice_{k}: {np.mean(mean_dice_cases[k])}')
        _log.info(f'mar_val batches meanIOU_{k}: {np.mean(mean_iou_cases[k])}') 
    
    # write validation result to log file
    m_meanDice = np.mean(mean_dice)
    m_meanPrec = np.mean(mean_prec)
    m_meanRec = np.mean(mean_rec)
    m_meanIOU = np.mean(mean_iou)

    _run.log_scalar('mar_val_batches_meanDice', m_meanDice)
    _run.log_scalar('mar_val_batches_meanPrec', m_meanPrec)
    _run.log_scalar('mar_val_al_batches_meanRec', m_meanRec)
    _run.log_scalar('mar_val_al_batches_meanIOU', m_meanIOU)
    _log.info(f'mar_val batches meanDice: {m_meanDice}')
    _log.info(f'mar_val batches meanPrec: {m_meanPrec}')
    _log.info(f'mar_val batches meanRec: {m_meanRec}')
    _log.info(f'mar_val batches meanIOU: {m_meanIOU}')
    print("============ ============")
    _log.info(f'End of validation')
    return 1
