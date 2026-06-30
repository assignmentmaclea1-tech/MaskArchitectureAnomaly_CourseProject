# Questo script valuta le performance di anomaly detection del modello EoMT su dataset
# stradali (RoadAnomaly21, LostAndFound, ecc.), usando metodi post-hoc per assegnare
# uno score di anomalia a ogni pixel. Il metodo principale è MSP con Temperature Scaling:
# i logits per-pixel vengono scalati da T prima della softmax, appiattendo la distribuzione
# e aumentando la separabilità tra pixel in-distribution e OOD.
# Le metriche calcolate sono AUPRC e FPR@TPR95, standard per l'anomaly segmentation.

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


def fpr_at_95_tpr(confidences, labels):
    # Calcola il False Positive Rate al punto in cui il True Positive Rate raggiunge 0.95.
    # È la metrica standard per valutare quanto "rumore" in-distribution viene classificato
    # come OOD quando il modello è tarato per catturare il 95% degli anomali reali.
    # argmax restituisce il primo indice in cui la condizione è vera lungo la curva ROC,
    # che corrisponde al threshold minimo per cui TPR >= 0.95.
    fpr, tpr, _ = roc_curve(labels, confidences)
    idx = np.argmax(tpr >= 0.95)
    return fpr[idx]

# Seed fisso per riproducibilità: garantisce che eventuali operazioni stocastiche
# (es. campionamento futuro) siano deterministiche tra run diverse.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# deterministic=True garantisce riproducibilità esatta su GPU a scapito di performance.
# benchmark=True è apparentemente in contraddizione (ottimizza i kernel cuDNN in modo
# adattivo), ma qui non causa non-determinismo perché il modello è in eval() e le
# operazioni sono fisse. In pratica benchmark=True accelera l'inference senza rompere
# la riproducibilità in questo contesto.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Le immagini vengono ridimensionate a (512, 1024): risoluzione standard Cityscapes.
# BILINEAR per le immagini (qualità visiva) e NEAREST per le maschere GT
# (evita di creare valori interpolati tra label discrete, es. 0.5 tra classe 0 e 1).
input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])

def build_eomt_model(config_path, ckpt_path, device):
    # Carica la configurazione YAML del modello e instanzia dinamicamente
    # encoder, network e LightningModule tramite class_path, evitando import
    # hardcoded. Questo rende il codice agnostico alla specifica variante di EoMT.
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    img_size = config.get("data", {}).get("init_args", {}).get("img_size", (640, 640))
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    num_classes = 19  # classi Cityscapes

    # Istanziazione dinamica dell'encoder tramite class_path nel config YAML.
    # rsplit(".", 1) separa "eomt.encoders.dinov2.DinoV2Encoder" in
    # ("eomt.encoders.dinov2", "DinoV2Encoder") per import dinamico.
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))

    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    # masked_attn_enabled=False: disabilita l'attention masking usato durante il training
    # per efficienza; a inference non è necessario e può causare inconsistenze.
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

    # Interpolazione del positional embedding: il checkpoint potrebbe essere stato
    # addestrato con una risoluzione diversa (es. 224x224 → griglia 14x14) rispetto
    # a quella usata qui (es. 640x640 → griglia 40x40). Si fa un resize 2D bicubico
    # del positional embedding per adattarlo alla nuova griglia senza perdere
    # la struttura spaziale appresa durante il pretraining.
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

    # strict=False permette di caricare checkpoint parziali: parametri non presenti
    # nel checkpoint vengono lasciati con i valori inizializzati dal costruttore,
    # e parametri extra nel checkpoint vengono ignorati silenziosamente.
    model.load_state_dict(state_dict, strict=False)
    return model

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp", nargs="+")
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--post_hoc', default="MSP", choices=["MSP", "MaxLogit", "MaxEntropy", "RbA"])
    parser.add_argument('--temperature', type=float, default=1.5, help="Parametro T per il Temperature Scaling")
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

        # Il modello richiede input uint8 [0, 255] invece che float [0, 1]:
        # la conversione avviene moltiplicando per 255 e castando a uint8.
        img_uint8 = (img_tensor * 255).to(torch.uint8)
        imgs = [img_uint8]
        img_sizes = [img.shape[-2:] for img in imgs]

        with torch.no_grad(), torch.autocast(dtype=torch.float16, device_type="cuda"):
            # window_imgs_semantic divide l'immagine in crops sovrapposti (windowing)
            # per gestire risoluzioni elevate che non entrerebbero in memoria come
            # singolo forward pass. origins contiene le coordinate per riassemblare.
            crops, origins = model.window_imgs_semantic(imgs)
            mask_logits_per_layer, class_logits_per_layer = model(crops)

            # Si prende solo l'ultimo layer del decoder: contiene le predizioni
            # finali, le più raffinate. I layer precedenti sono auxiliary losses
            # usati solo durante il training.
            mask_logits = F.interpolate(mask_logits_per_layer[-1], crops.shape[-2:], mode="bilinear", align_corners=False)
            class_logits = class_logits_per_layer[-1]

            # to_per_pixel_logits_semantic combina mask_logits e class_logits
            # in logits per-pixel per ciascuna delle 19 classi, tramite prodotto
            # tra le maschere binarie e i vettori di classe delle query.
            crop_logits = model.to_per_pixel_logits_semantic(mask_logits, class_logits)

            # revert_window_logits_semantic riassembla i crop nella risoluzione
            # originale dell'immagine, mediando le zone di sovrapposizione.
            # [0] perché il batch size è 1.
            logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)[0]
            # logits ha shape (num_classes=19, H, W)

            # Temperature Scaling: dividere per T > 1 appiattisce la distribuzione
            # softmax, aumentando l'incertezza del modello sui pixel in-distribution
            # e migliorando la separabilità con i pixel OOD.
            scaled_logits = logits / args.temperature
            probs = F.softmax(scaled_logits, dim=0)

            if args.post_hoc == "MSP":
                # Score di anomalia = 1 - probabilità massima: pixel su cui il modello
                # è poco sicuro (max prob basso) ricevono score alto → più anomali.
                anomaly_result = 1.0 - probs.max(dim=0)[0].cpu().numpy()
            elif args.post_hoc == "MaxLogit":
                # Usa scaled_logits invece dei logits originali: la temperatura
                # influenza anche questo metodo coerentemente con MSP.
                anomaly_result = -scaled_logits.max(dim=0)[0].cpu().numpy()
            elif args.post_hoc == "MaxEntropy":
                # Entropia della distribuzione: valori alti = distribuzione piatta = incertezza alta.
                # Il segno positivo è corretto: entropia alta corrisponde a score anomalia alto.
                # Nota: nella versione precedente del codice c'era un '-' di troppo qui.
                anomaly_result = torch.sum(probs * torch.log(probs + 1e-8), dim=0).cpu().numpy()
            elif args.post_hoc == "RbA":
                # RbA (Residual by Aggregation): usa la probabilità dell'ultima classe
                # come proxy per l'anomalia. Funziona se il modello è stato fine-tuned
                # con una classe "void" o "unknown" esplicita come ultima.
                anomaly_result = probs[-1, :, :].cpu().numpy()

        # Costruzione del path della GT: la struttura attesa è che le immagini
        # siano in .../images/... e le maschere in .../labels_masks/...
        # con la stessa struttura e nome file, cambiando solo l'estensione.
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

        # Rimappatura dataset-specifica delle label GT a valori binari:
        # 0 = in-distribution, 1 = OOD/anomalia, 255 = ignored (void).
        # Ogni dataset usa convenzioni diverse nei valori delle maschere.
        if "RoadAnomaly" in pathGT:
             ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
             # In LostAndFound: 0=background(void), 1=road(in-dist), 2-200=obstacles(OOD)
             ood_gts = np.where((ood_gts==0), 255, ood_gts)   # background → ignore
             ood_gts = np.where((ood_gts==1), 0, ood_gts)     # road → in-distribution
             ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)  # ostacoli → OOD
        if "Streethazard" in pathGT:
             ood_gts = np.where((ood_gts==14), 255, ood_gts)  # classe anomala originale → ignore temporaneo
             ood_gts = np.where((ood_gts<20), 0, ood_gts)     # classi note → in-distribution
             ood_gts = np.where((ood_gts==255), 1, ood_gts)   # anomalie → OOD

        # Immagini senza pixel OOD (label 1 assente) vengono scartate:
        # non contribuiscono al calcolo delle metriche e potrebbero distorcere
        # la stima di FPR (solo falsi positivi, nessun vero positivo).
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

    # Appiattimento: ood_gts e anomaly_scores diventano array 3D (N, H, W),
    # poi le maschere booleane selezionano tutti i pixel OOD e in-dist
    # indipendentemente dall'immagine di appartenenza.
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)
    # I pixel con label 255 (void/ignore) vengono implicitamente esclusi
    # non essendo né in ood_mask né in ind_mask.

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'Method: {args.post_hoc} | AUPRC score: {prc_auc*100.0:.2f} | FPR@TPR95: {fpr*100.0:.2f}')

if __name__ == '__main__':
    main()
