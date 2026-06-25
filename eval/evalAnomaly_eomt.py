#The results are summarized in results.txt
import sys
import os
import cv2
import glob
import torch
import random
import yaml
import math
import importlib
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor
import torch.nn.functional as F

sys.path.append("/content/MaskArchitectureAnomaly_CourseProject/eomt")

def fpr_at_95_tpr(confidences, labels):
    fpr, tpr, _ = roc_curve(labels, confidences)
    idx = np.argmax(tpr >= 0.95)
    return fpr[idx]

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])

def build_eomt_model(config_path, ckpt_path, device):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    img_size = config.get("data", {}).get("init_args", {}).get("img_size", (640, 640))
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    num_classes = 19

    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=num_classes,
        encoder=encoder,
        **network_kwargs,
    )

    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}

    model = lit_cls(
        img_size=img_size,
        num_classes=num_classes,
        network=network,
        **model_kwargs,
    )

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)

    pos_key = 'network.encoder.backbone.pos_embed'
    if pos_key in state_dict:
        ckpt_pos = state_dict[pos_key]
        model_pos = model.state_dict()[pos_key]

        if ckpt_pos.shape != model_pos.shape:
            C = ckpt_pos.shape[-1]
            grid_ckpt = int(math.sqrt(ckpt_pos.shape[1]))
            grid_model = int(math.sqrt(model_pos.shape[1]))

            ckpt_pos_2d = ckpt_pos.reshape(1, grid_ckpt, grid_ckpt, C).permute(0, 3, 1, 2)
            model_pos_2d = F.interpolate(ckpt_pos_2d, size=(grid_model, grid_model), mode='bicubic', align_corners=False)
            state_dict[pos_key] = model_pos_2d.permute(0, 2, 3, 1).reshape(1, grid_model * grid_model, C)

    model.load_state_dict(state_dict, strict=False)
    return model

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp", nargs="+")
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--post_hoc', default="MSP", choices=["MSP", "MaxLogit", "MaxEntropy", "RbA"])
    parser.add_argument('--subset', default="val")
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    device = "cpu" if args.cpu else "cuda"
    model = build_eomt_model(args.config, args.ckpt_path, device)
    if not args.cpu:
        model = model.cuda()
    model.eval()
    print("Model LOADED successfully!")

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(f"Processing: {os.path.basename(path)}")
        img_tensor = input_transform((Image.open(path).convert('RGB'))).cuda()

        img_uint8 = (img_tensor * 255).to(torch.uint8)
        imgs = [img_uint8]
        img_sizes = [img.shape[-2:] for img in imgs]

        with torch.no_grad(), torch.autocast(dtype=torch.float16, device_type="cuda"):
            crops, origins = model.window_imgs_semantic(imgs)
            mask_logits_per_layer, class_logits_per_layer = model(crops)

            mask_logits = F.interpolate(mask_logits_per_layer[-1], crops.shape[-2:], mode="bilinear", align_corners=False)
            class_logits = class_logits_per_layer[-1]

            crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)
            logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)[0]

            probs = F.softmax(logits, dim=0)

            if args.post_hoc == "MSP":
                anomaly_result = 1.0 - probs.max(dim=0)[0].cpu().numpy()
            elif args.post_hoc == "MaxLogit":
                anomaly_result = -logits.max(dim=0)[0].cpu().numpy()
            elif args.post_hoc == "MaxEntropy":
                anomaly_result = -torch.sum(probs * torch.log(probs + 1e-8), dim=0).cpu().numpy()
            elif args.post_hoc == "RbA":
                anomaly_result = probs[-1, :, :].cpu().numpy()

        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
             ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
             ood_gts = np.where((ood_gts==0), 255, ood_gts)
             ood_gts = np.where((ood_gts==1), 0, ood_gts)
             ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
        if "Streethazard" in pathGT:
             ood_gts = np.where((ood_gts==14), 255, ood_gts)
             ood_gts = np.where((ood_gts<20), 0, ood_gts)
             ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)

        del mask_logits_per_layer, class_logits_per_layer, logits, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write("\n")

    if len(ood_gts_list) == 0:
        print("Nessuna immagine valida trovata per il calcolo delle metriche.")
        return

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'Method: {args.post_hoc} | AUPRC score: {prc_auc*100.0:.2f} | FPR@TPR95: {fpr*100.0:.2f}')

    file.write((f'Method: {args.post_hoc} | AUPRC score: {prc_auc*100.0:.2f} | FPR@TPR95: {fpr*100.0:.2f}'))
    file.close()

if __name__ == '__main__':
    main()
