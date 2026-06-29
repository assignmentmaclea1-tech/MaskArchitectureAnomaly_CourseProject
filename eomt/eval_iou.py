# iouEval_eomt.py
# Calcola IoU (mean e per-class) su un dataset custom usando EoMT.
# La temperatura viene passata direttamente a per_pixel_maps,
# che la applica ai logit di classe prima del softmax.
#
# Esempio:
#   cd /content/MaskArchitectureAnomaly_CourseProject/eomt
#   python iouEval_eomt.py \
#       --input "/content/drive/MyDrive/dataset/images" \
#       --gt_dir "/content/drive/MyDrive/dataset/labels" \
#       --ckpt_path "/content/drive/MyDrive/pesi/eomt_cityscapes.bin" \
#       --config "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml" \
#       --temperature 1.5

import os
import glob
import time
from argparse import ArgumentParser

import torch.nn.functional as F
from torch.amp.autocast_mode import autocast

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

from evalAnomaly_eomt import build_eomt_model
from iouEval import iouEval, getColorEntry

NUM_CLASSES = 20  # 19 classi Cityscapes + 1 ignore


# ============================================================================
# DATASET CUSTOM
# Restituisce coppie (immagine, maschera GT) da due cartelle separate.
# Si aspetta che immagine e maschera abbiano lo stesso nome file,
# con estensioni potenzialmente diverse (es. .png / .png).
# ============================================================================
class SegmentationDataset(torch.utils.data.Dataset):
    """
    Dataset minimale per segmentazione semantica.
    Cerca le immagini in img_dir e le maschere GT in gt_dir,
    abbinandole per nome file (senza estensione).
    """
    def __init__(self, img_dir, gt_dir, img_size=(512, 1024)):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.gt_dir = gt_dir
        self.img_size = img_size

        # Trasformazione immagine: resize + conversione in tensore float [0,1]
        self.img_transform = Compose([
            Resize(img_size, Image.BILINEAR),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        filename = os.path.splitext(os.path.basename(img_path))[0]

        # Carica immagine RGB
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.img_transform(img)  # [3, H, W] float [0,1]

        # Cerca la maschera GT con lo stesso nome (qualsiasi estensione)
        gt_candidates = glob.glob(os.path.join(self.gt_dir, filename + ".*"))
        if len(gt_candidates) == 0:
            raise FileNotFoundError(f"Maschera GT non trovata per: {filename}")
        gt_path = gt_candidates[0]

        # Carica GT: resize con NEAREST per non alterare le etichette
        gt = Image.open(gt_path)
        gt = gt.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        gt_tensor = torch.from_numpy(np.array(gt)).long()  # [H, W]

        # Rimappa etichetta 255 (ignore) a 19
        gt_tensor[gt_tensor == 255] = 19

        return img_tensor, gt_tensor, filename


def per_pixel_maps(model, crops, origins, img_sizes, temperature):
    """
    Esegue il forward pass e ricostruisce due mappe per-pixel alla risoluzione
    originale dell'immagine, necessarie per calcolare tutti i metodi post-hoc.

    Il decoder EoMT produce Q maschere binarie e Q vettori di classe.
    Per ottenere logit per-pixel si combina:
      - sigmoid(mask_logit)  →  peso spaziale di ogni query [Q, H, W]
      - softmax(class_logit) →  distribuzione di classe di ogni query [Q, C]
    tramite prodotto einsum: S[c,h,w] = Σ_q mask[q,h,w] * class_soft[q,c]

    Si calcolano due mappe distinte:
      S: basata su softmax(class/T)  → usata da MSP, MaxEntropy, RbA e IoU
      L: basata su class_logit grezzo → usata da MaxLogit

    Args:
        model:       modello EoMT in eval mode.
        crops:       crops estratti da window_imgs_semantic [N, C, h, w].
        origins:     coordinate di origine di ogni crop nella scena.
        img_sizes:   dimensioni originali delle immagini.
        temperature: valore T per scalare i logit di classe prima del softmax.
                     T > 1 ammorbidisce la distribuzione (più incertezza).
                     T < 1 la aguzza (più confidenza).
                     T = 1 nessuna modifica (comportamento standard).

    Returns:
        S: mappa probabilità per-pixel [C, H, W] (softmax pesata per maschera).
        L: mappa logit grezzi per-pixel [C, H, W].
    """
    with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
        mask_logits_per_layer, class_logits_per_layer = model(crops)

    # Si prende solo l'output dell'ultimo layer del decoder
    mask_logits  = mask_logits_per_layer[-1].float()   # [N, Q, h, w]
    class_logits = class_logits_per_layer[-1].float()  # [N, Q, C+1]

    # Interpola le maschere alla risoluzione interna del modello (img_size del crop)
    mask_logits = F.interpolate(mask_logits, model.img_size, mode="bilinear")

    mask_p   = mask_logits.sigmoid()                               # [N, Q, h, w]: peso spaziale
    cls_soft = (class_logits / temperature).softmax(dim=-1)[..., :-1]  # [N, Q, C]: drop no-object
    cls_raw  = class_logits[..., :-1]                             # [N, Q, C]: logit grezzi

    # Per ogni pixel, somma pesata sulle Q queries tramite einsum
    S_crop = torch.einsum("nqhw, nqc -> nchw", mask_p, cls_soft)  # [N, C, h, w]
    L_crop = torch.einsum("nqhw, nqc -> nchw", mask_p, cls_raw)   # [N, C, h, w]

    # Riassembla i crops nella risoluzione originale dell'immagine
    S = model.revert_window_logits_semantic(S_crop, origins, img_sizes)[0]  # [C, H, W]
    L = model.revert_window_logits_semantic(L_crop, origins, img_sizes)[0]  # [C, H, W]
    return S, L



# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = ArgumentParser()
    parser.add_argument("--input",      required=True,
                        help="Cartella contenente le immagini da valutare")
    parser.add_argument("--gt_dir",     required=True,
                        help="Cartella contenente le maschere GT")
    parser.add_argument("--ckpt_path",  required=True,
                        help="Percorso al file dei pesi .bin di EoMT")
    parser.add_argument("--config",     required=True,
                        help="Percorso al file YAML di configurazione EoMT")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperatura per il scaling dei logit (default: 1.0 = nessuno scaling)")
    parser.add_argument("--cpu",        action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"

    # Carica il modello EoMT con i pesi locali
    model = build_eomt_model(args.config, args.ckpt_path, device)
    model = model.cuda()
    model.eval()
    print("Modello caricato con successo.")

    # Dataset e DataLoader (batch_size=1 perché le immagini possono avere risoluzioni diverse)
    dataset = SegmentationDataset(args.input, args.gt_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    iouEvalVal = iouEval(NUM_CLASSES)
    start = time.time()

    for img_tensor, gt_tensor, filename in loader:
        print(f"Immagine: {filename[0]}")
        print(f"img_tensor shape: {img_tensor.shape}")
        print(f"gt_tensor shape: {gt_tensor.shape}")
        print(f"GT valori unici: {gt_tensor.unique()}")
    
        img_uint8 = (img_tensor.squeeze(0) * 255).to(torch.uint8).to(device)
        imgs = [img_uint8]
        img_sizes = [img_uint8.shape[-2:]]
    
        crops, origins = model.window_imgs_semantic(imgs)
        S, L = per_pixel_maps(model, crops, origins, img_sizes, args.temperature)
    
        print(f"S shape: {S.shape}")
        print(f"S valori unici (primi 5): {S.unique()[:5]}")
    
        pred = S.argmax(dim=0).unsqueeze(0).unsqueeze(0)
        print(f"pred shape: {pred.shape}")
        print(f"pred valori unici: {pred.unique()}")
    
        gt = gt_tensor.unsqueeze(1).to(device)
        print(f"gt shape: {gt.shape}")
    
        iouEvalVal.addBatch(pred.data, gt.data)

        del S, L, crops
        if device == "cuda":
            torch.cuda.empty_cache()

    # Calcolo e stampa metriche finali
    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i]) + '{:0.2f}'.format(iou_classes[i] * 100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print(f"Tempo totale: {time.time() - start:.1f} secondi")
    print("=======================================")
    print("Per-Class IoU:")
    print(iou_classes_str[0],  "Road")
    print(iou_classes_str[1],  "Sidewalk")
    print(iou_classes_str[2],  "Building")
    print(iou_classes_str[3],  "Wall")
    print(iou_classes_str[4],  "Fence")
    print(iou_classes_str[5],  "Pole")
    print(iou_classes_str[6],  "Traffic light")
    print(iou_classes_str[7],  "Traffic sign")
    print(iou_classes_str[8],  "Vegetation")
    print(iou_classes_str[9],  "Terrain")
    print(iou_classes_str[10], "Sky")
    print(iou_classes_str[11], "Person")
    print(iou_classes_str[12], "Rider")
    print(iou_classes_str[13], "Car")
    print(iou_classes_str[14], "Truck")
    print(iou_classes_str[15], "Bus")
    print(iou_classes_str[16], "Train")
    print(iou_classes_str[17], "Motorcycle")
    print(iou_classes_str[18], "Bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal) + '{:0.2f}'.format(iouVal * 100) + '\033[0m'
    print("MEAN IoU: ", iouStr, "%")


if __name__ == "__main__":
    main()
