%%writefile /content/MaskArchitectureAnomaly_CourseProject/eomt/evalAnomaly_eomt.py
# evalAnomaly_eomt.py
# Anomaly segmentation evaluation for the EoMT (mask architecture) model.
# Da lanciare DALLA cartella eomt/ (serve per importare models. / training.).
#
# Esempio:
#   cd /content/MaskArchitectureAnomaly_CourseProject/eomt
#   python evalAnomaly_eomt.py \
#       --input "/content/drive/MyDrive/Anomaly_Validation_Datasets/Validation_Dataset/RoadAnomaly21/images/*.png" \
#       --temperature 1.0
#
# Stampa AuPRC e FPR@95 per TUTTI i metodi (MSP, MaxLogit, MaxEntropy, RbA) in un solo forward pass.

import os
import glob
import yaml
import random
import importlib
import warnings
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from PIL import Image
from huggingface_hub import hf_hub_download

from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score

# ----------------------------- reproducibility -----------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# ============================================================================
# 1. CARICAMENTO DEL MODELLO EoMT (mirror di inference.ipynb, celle 6 + 8)
# ============================================================================
def load_eomt(config_path, hf_name, img_size, num_classes, device):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    warnings.filterwarnings("ignore", message=r".*is already saved during checkpointing.*")

    # --- encoder (ViT / DINOv2) ---
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_cls = encoder_cfg["class_path"].rsplit(".", 1)
    encoder = getattr(importlib.import_module(enc_mod), enc_cls)(
        img_size=img_size, **encoder_cfg.get("init_args", {})
    )

    # --- network EoMT ---
    net_cfg = config["model"]["init_args"]["network"]
    net_mod, net_cls = net_cfg["class_path"].rsplit(".", 1)
    net_kwargs = {k: v for k, v in net_cfg["init_args"].items() if k != "encoder"}
    network = getattr(importlib.import_module(net_mod), net_cls)(
        masked_attn_enabled=False,      # in inference disattiviamo la masked attention
        num_classes=num_classes,
        encoder=encoder,
        **net_kwargs,                   # num_q=100, num_blocks=3
    )

    # --- Lightning module (porta con sé i metodi di windowing/post-process) ---
    lit_mod, lit_cls = config["model"]["class_path"].rsplit(".", 1)
    model = (
        getattr(importlib.import_module(lit_mod), lit_cls)(
            img_size=img_size,
            num_classes=num_classes,
            network=network,
            attn_mask_annealing_enabled=False,   # arg obbligatorio
        )
        .eval()
        .to(device)
    )

    # --- pesi pre-addestrati da HuggingFace (NON esiste un .pth locale) ---
    state_dict_path = hf_hub_download(repo_id=f"tue-mps/{hf_name}", filename="pytorch_model.bin")
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    missing = model.load_state_dict(state_dict, strict=False)
    print(f"Pesi caricati da tue-mps/{hf_name} | missing={len(missing.missing_keys)} "
          f"unexpected={len(missing.unexpected_keys)}")
    return model


# ============================================================================
# 2. DA OUTPUT EoMT (maschere + classi) A MAPPE PER-PIXEL [C, H, W]
# ============================================================================
def per_pixel_maps(model, crops, origins, img_sizes, temperature):
    """Ritorna due mappe a risoluzione originale:
       S = sigmoid(mask) . softmax(class/T)[:-1]   (probabilita' -> MSP/Entropy/RbA)
       L = sigmoid(mask) . class[:-1]              (logit grezzi -> MaxLogit)
    """
    with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
        mask_logits_per_layer, class_logits_per_layer = model(crops)

    mask_logits = mask_logits_per_layer[-1].float()      # [N, Q, h, w]
    class_logits = class_logits_per_layer[-1].float()    # [N, Q, C+1]
    mask_logits = F.interpolate(mask_logits, model.img_size, mode="bilinear")

    mask_p = mask_logits.sigmoid()                                   # [N, Q, h, w]
    cls_soft = (class_logits / temperature).softmax(dim=-1)[..., :-1]  # [N, Q, C]  (drop no-object)
    cls_raw = class_logits[..., :-1]                                  # [N, Q, C]

    S_crop = torch.einsum("nqhw, nqc -> nchw", mask_p, cls_soft)     # [N, C, h, w]
    L_crop = torch.einsum("nqhw, nqc -> nchw", mask_p, cls_raw)      # [N, C, h, w]

    # ricombina le finestre (crop) nella risoluzione originale dell'immagine
    S = model.revert_window_logits_semantic(S_crop, origins, img_sizes)[0]  # [C, H, W]
    L = model.revert_window_logits_semantic(L_crop, origins, img_sizes)[0]  # [C, H, W]
    return S, L


def anomaly_scores(S, L):
    """Calcola i 4 anomaly score (numpy [H, W]) dalle mappe per-pixel."""
    p = S / S.sum(0, keepdim=True).clamp_min(1e-8)        # normalizza a distribuzione
    return {
        "MSP":        (1.0 - S.max(0).values).cpu().numpy(),
        "MaxLogit":   (-L.max(0).values).cpu().numpy(),
        "MaxEntropy": (-(p * p.clamp_min(1e-8).log()).sum(0)).cpu().numpy(),
        "RbA":        (-S.sum(0)).cpu().numpy(),
    }


# ============================================================================
# 3. GROUND TRUTH (stesso remapping per-dataset dello script ERFNet)
# ============================================================================
def load_gt(path):
    pathGT = path.replace("images", "labels_masks")
    if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
    if "fs_static" in pathGT:      pathGT = pathGT.replace("jpg", "png")
    if "RoadAnomaly" in pathGT:    pathGT = pathGT.replace("jpg", "png")

    ood_gts = np.array(Image.open(pathGT))
    if "RoadAnomaly" in pathGT:
        ood_gts = np.where((ood_gts == 2), 1, ood_gts)
    if "LostAndFound" in pathGT:
        ood_gts = np.where((ood_gts == 0), 255, ood_gts)
        ood_gts = np.where((ood_gts == 1), 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
    if "Streethazard" in pathGT:
        ood_gts = np.where((ood_gts == 14), 255, ood_gts)
        ood_gts = np.where((ood_gts < 20), 0, ood_gts)
        ood_gts = np.where((ood_gts == 255), 1, ood_gts)
    return ood_gts


# ============================================================================
# 4. MAIN
# ============================================================================
def main():
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+",
                        help="glob delle immagini, es. '.../RoadAnomaly21/images/*.png'")
    parser.add_argument("--config", default="configs/dinov2/cityscapes/semantic/eomt_base_640.yaml")
    parser.add_argument("--hf_name", default="cityscapes_semantic_eomt_base_640",
                        help="repo HuggingFace dei pesi (se base_640 non esiste, prova cityscapes_semantic_eomt_large_1024)")
    parser.add_argument("--img_size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_eomt(args.config, args.hf_name, tuple(args.img_size), args.num_classes, device)

    methods = ["MSP", "MaxLogit", "MaxEntropy", "RbA"]
    # accumulo per-immagine (le immagini hanno risoluzioni diverse -> non si possono impilare)
    in_scores = {m: [] for m in methods}
    ood_scores = {m: [] for m in methods}

    for path in sorted(glob.glob(os.path.expanduser(str(args.input[0])))):
        ood_gts = load_gt(path)
        if 1 not in np.unique(ood_gts):
            continue  # immagine senza anomalie -> inutile

        # input come se ne facesse parte il dataset EoMT: uint8 [0,255], C,H,W (NIENTE ToTensor/[0,1])
        img = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).contiguous()
        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:]]
        crops, origins = model.window_imgs_semantic(imgs)

        S, L = per_pixel_maps(model, crops, origins, img_sizes, args.temperature)
        scores = anomaly_scores(S, L)  # ogni score e' [H, W] alla risoluzione originale

        in_mask = (ood_gts == 0)
        ood_mask = (ood_gts == 1)
        for m in methods:
            in_scores[m].append(scores[m][in_mask])
            ood_scores[m].append(scores[m][ood_mask])

        del S, L, crops
        if device == "cuda":
            torch.cuda.empty_cache()

    # ----------------------------- metriche -----------------------------
    print(f"\n=== Risultati (T={args.temperature}) ===")
    for m in methods:
        val_out = np.concatenate(in_scores[m] + ood_scores[m])
        val_label = np.concatenate(
            [np.zeros(len(a)) for a in in_scores[m]] + [np.ones(len(a)) for a in ood_scores[m]]
        )
        auprc = average_precision_score(val_label, val_out) * 100.0
        fpr = fpr_at_95_tpr(val_out, val_label) * 100.0
        print(f"{m:11s} | AuPRC: {auprc:6.2f} | FPR@95: {fpr:6.2f}")


if __name__ == "__main__":
    main()
