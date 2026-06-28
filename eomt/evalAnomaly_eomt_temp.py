#%%writefile /content/MaskArchitectureAnomaly_CourseProject/eomt/evalAnomaly_eomt.py
# Esempio:
#   cd /content/MaskArchitectureAnomaly_CourseProject/eomt
#   python evalAnomaly_eomt.py \
#       --input "/content/drive/MyDrive/Anomaly_Validation_Datasets/Validation_Dataset/RoadAnomaly21/images/*.png" \
#       --temperature 1.0
# Questo codice serve per valutare la Temperature Scaling.
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

# Fissa tutti i seed per garantire riproducibilità tra run diverse
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# ============================================================================
# 1. CARICAMENTO DEL MODELLO
# ============================================================================

def load_eomt(config_path, hf_name, img_size, num_classes, device):
    """
    Costruisce il modello EoMT da config YAML e ne carica i pesi da HuggingFace Hub.

    La costruzione segue tre passi, ognuno istanziato dinamicamente
    tramite il class_path definito nel YAML:
      1. Encoder (es. DINOv2/ViT) con la img_size di destinazione.
      2. Rete EoMT completa (encoder + decoder a maschere).
      3. Lightning Module che aggiunge windowing e post-processing.

    I pesi vengono scaricati automaticamente dal repo 'tue-mps/{hf_name}'
    su HuggingFace; non è richiesto nessun checkpoint locale.

    Args:
        config_path: percorso al file YAML del modello.
        hf_name:     nome del repo HuggingFace (es. 'cityscapes_semantic_eomt_base_640').
        img_size:    tupla (H, W) attesa dal modello (es. (640, 640)).
        num_classes: numero di classi semantiche (19 per Cityscapes).
        device:      'cuda' o 'cpu'.

    Returns:
        model: modello EoMT in modalità eval, pronto per l'inferenza.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Sopprime warning di Lightning sui checkpoint che non impattano l'inferenza
    warnings.filterwarnings("ignore", message=r".*is already saved during checkpointing.*")

    # Istanziazione dinamica dell'encoder: il class_path (es. "eomt.encoders.vit.ViT")
    # viene splittato per importare il modulo e recuperare la classe via importlib
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_cls = encoder_cfg["class_path"].rsplit(".", 1)
    encoder = getattr(importlib.import_module(enc_mod), enc_cls)(
        img_size=img_size, **encoder_cfg.get("init_args", {})
    )

    # Istanziazione della rete EoMT: si rimuove 'encoder' dai kwargs
    # perché viene passato esplicitamente come argomento separato
    net_cfg = config["model"]["init_args"]["network"]
    net_mod, net_cls = net_cfg["class_path"].rsplit(".", 1)
    net_kwargs = {k: v for k, v in net_cfg["init_args"].items() if k != "encoder"}
    network = getattr(importlib.import_module(net_mod), net_cls)(
        masked_attn_enabled=False,  # la masked attention è usata solo in training
        num_classes=num_classes,
        encoder=encoder,
        **net_kwargs,
    )

    # Istanziazione del Lightning Module: wrappa la rete e fornisce
    # i metodi window_imgs_semantic / revert_window_logits_semantic
    lit_mod, lit_cls = config["model"]["class_path"].rsplit(".", 1)
    model = (
        getattr(importlib.import_module(lit_mod), lit_cls)(
            img_size=img_size,
            num_classes=num_classes,
            network=network,
            attn_mask_annealing_enabled=False,
        )
        .eval()
        .to(device)
    )

    # Download dei pesi da HuggingFace Hub (nessun file locale richiesto)
    state_dict_path = hf_hub_download(repo_id=f"tue-mps/{hf_name}", filename="pytorch_model.bin")
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    # strict=False tollera chiavi mancanti/extra dovute a versioni leggermente diverse
    missing = model.load_state_dict(state_dict, strict=False)
    print(f"Pesi caricati da tue-mps/{hf_name} | missing={len(missing.missing_keys)} "
          f"unexpected={len(missing.unexpected_keys)}")
    return model


# ============================================================================
# 2. MAPPE PER-PIXEL E ANOMALY SCORE
# ============================================================================

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
      S: basata su softmax(class/T)  → usata da MSP, MaxEntropy, RbA
      L: basata su class_logit grezzo → usata da MaxLogit

    Args:
        model:       modello EoMT in eval mode.
        crops:       crops estratti da window_imgs_semantic [N, C, h, w].
        origins:     coordinate di origine di ogni crop nella scena.
        img_sizes:   dimensioni originali delle immagini.
        temperature: valore T per scalare i logit di classe prima del softmax
                     (T > 1 ammorbidisce la distribuzione, T < 1 la aguzza).

    Returns:
        S: mappa probabilità per-pixel [C, H, W] (somma ≈ 1 per pixel).
        L: mappa logit grezzi per-pixel [C, H, W].
    """
    with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
        mask_logits_per_layer, class_logits_per_layer = model(crops)

    # Si prende solo l'output dell'ultimo layer del decoder
    mask_logits = mask_logits_per_layer[-1].float()   # [N, Q, h, w]
    class_logits = class_logits_per_layer[-1].float() # [N, Q, C+1]

    # Interpola le maschere alla risoluzione interna del modello (img_size del crop)
    mask_logits = F.interpolate(mask_logits, model.img_size, mode="bilinear")

    mask_p = mask_logits.sigmoid()                              # [N, Q, h, w]: peso spaziale
    cls_soft = (class_logits / temperature).softmax(dim=-1)[..., :-1]  # [N, Q, C]: drop no-object
    cls_raw  = class_logits[..., :-1]                          # [N, Q, C]: logit grezzi

    # Combinazione: per ogni pixel, somma pesata sulle Q queries
    S_crop = torch.einsum("nqhw, nqc -> nchw", mask_p, cls_soft)  # [N, C, h, w]
    L_crop = torch.einsum("nqhw, nqc -> nchw", mask_p, cls_raw)   # [N, C, h, w]

    # Riassembla i crops nella risoluzione originale dell'immagine
    S = model.revert_window_logits_semantic(S_crop, origins, img_sizes)[0]  # [C, H, W]
    L = model.revert_window_logits_semantic(L_crop, origins, img_sizes)[0]  # [C, H, W]
    return S, L


def anomaly_scores(S, L):
    """
    Calcola i 4 anomaly score per ogni pixel a partire dalle mappe S e L.

    Tutti i metodi assegnano score più alto ai pixel anomali (OOD):
      MSP:        1 - max_c(S[c])         → bassa confidenza = OOD
      MaxLogit:   -max_c(L[c])            → logit basso = OOD
      MaxEntropy: -Σ_c p[c] * log(p[c])  → alta entropia = OOD
      RbA:        -Σ_c S[c]              → bassa massa totale = OOD (pixel non assegnati)

    Args:
        S: mappa probabilità [C, H, W] (softmax pesata per maschera).
        L: mappa logit grezzi [C, H, W].

    Returns:
        dizionario {nome_metodo: array numpy [H, W]}.
    """
    # Normalizza S a distribuzione valida (la somma su C potrebbe non essere esattamente 1
    # dopo il windowing, quindi si riscala per sicurezza)
    p = S / S.sum(0, keepdim=True).clamp_min(1e-8)

    return {
        "MSP":        (1.0 - S.max(0).values).cpu().numpy(),
        "MaxLogit":   (-L.max(0).values).cpu().numpy(),
        "MaxEntropy": (-(p * p.clamp_min(1e-8).log()).sum(0)).cpu().numpy(),
        "RbA":        (-S.sum(0)).cpu().numpy(),
    }


# ============================================================================
# 3. GROUND TRUTH
# ============================================================================

def load_gt(path):
    """
    Carica e binarizza la maschera GT per una data immagine.

    Ogni dataset usa convenzioni diverse per le etichette, quindi
    si applica un remapping specifico per ottenere sempre:
      0   → pixel in-distribution (inlier)
      1   → pixel OOD (anomalia)
      255 → pixel da ignorare nel calcolo delle metriche

    Struttura attesa delle cartelle:
      .../images/xxx.ext  →  immagine
      .../labels_masks/xxx.png  →  maschera GT

    Args:
        path: percorso all'immagine di input.

    Returns:
        ood_gts: array numpy [H, W] con valori in {0, 1, 255}.
    """
    # Costruisce il percorso della GT sostituendo la cartella e l'estensione
    pathGT = path.replace("images", "labels_masks")
    if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
    if "fs_static"      in pathGT: pathGT = pathGT.replace("jpg",  "png")
    if "RoadAnomaly"    in pathGT: pathGT = pathGT.replace("jpg",  "png")

    ood_gts = np.array(Image.open(pathGT))

    # RoadAnomaly: classe 2 = OOD (0 = road, 1 = inlier generico)
    if "RoadAnomaly" in pathGT:
        ood_gts = np.where((ood_gts == 2), 1, ood_gts)

    # LostAndFound: 0 = ignora (sfondo), 1 = inlier, 2-200 = OOD
    if "LostAndFound" in pathGT:
        ood_gts = np.where((ood_gts == 0), 255, ood_gts)
        ood_gts = np.where((ood_gts == 1), 0,   ood_gts)
        ood_gts = np.where((ood_gts >  1) & (ood_gts < 201), 1, ood_gts)

    # StreetHazards: 14 = ignora, <20 = inlier, 255 = OOD
    if "Streethazard" in pathGT:
        ood_gts = np.where((ood_gts == 14), 255, ood_gts)
        ood_gts = np.where((ood_gts <  20), 0,   ood_gts)
        ood_gts = np.where((ood_gts == 255), 1,  ood_gts)

    return ood_gts


# ============================================================================
# 4. MAIN
# ============================================================================

def main():
    """
    Entry point. Gestisce:
      - Parsing degli argomenti CLI.
      - Caricamento del modello da HuggingFace.
      - Loop di inferenza immagine per immagine.
      - Accumulo degli score per pixel in-distribution e OOD.
      - Calcolo e stampa di AuPRC e FPR@95 per tutti i metodi.

    Le immagini vengono processate singolarmente (non in batch) perché
    i dataset di anomalia hanno risoluzioni eterogenee che non permettono
    di impilare i tensori in un unico batch.
    """
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+",
                        help="glob delle immagini, es. '.../RoadAnomaly21/images/*.png'")
    parser.add_argument("--config", default="configs/dinov2/cityscapes/semantic/eomt_base_640.yaml")
    parser.add_argument("--hf_name", default="cityscapes_semantic_eomt_base_640",
                        help="repo HuggingFace (es. cityscapes_semantic_eomt_large_1024)")
    parser.add_argument("--img_size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--num_classes", type=int, default=19)
    # Temperature scaling: valori > 1 ammorbidiscono il softmax e possono
    # migliorare la calibrazione OOD su alcuni dataset
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_eomt(args.config, args.hf_name, tuple(args.img_size), args.num_classes, device)

    methods = ["MSP", "MaxLogit", "MaxEntropy", "RbA"]

    # Accumulo separato per pixel in-distribution e OOD:
    # si usano liste di array (non un unico array) perché ogni immagine
    # può avere risoluzione e numero di pixel OOD diversi
    in_scores  = {m: [] for m in methods}
    ood_scores = {m: [] for m in methods}

    for path in sorted(glob.glob(os.path.expanduser(str(args.input[0])))):
        ood_gts = load_gt(path)

        # Salta le immagini senza pixel OOD: non contribuiscono alle metriche
        if 1 not in np.unique(ood_gts):
            continue

        # Il modello EoMT si aspetta uint8 [0,255] con shape [C, H, W]:
        # non si usa ToTensor (che normalizza in [0,1]), ma una conversione diretta
        img = torch.from_numpy(
            np.array(Image.open(path).convert("RGB"))
        ).permute(2, 0, 1).contiguous()

        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:]]

        # Divide l'immagine in finestre sovrapposte (crops) gestendo le risoluzioni arbitrarie
        crops, origins = model.window_imgs_semantic(imgs)

        S, L = per_pixel_maps(model, crops, origins, img_sizes, args.temperature)
        scores = anomaly_scores(S, L)  # ogni score è [H, W] alla risoluzione originale

        # Separa i pixel in-distribution (0) da quelli OOD (1)
        # I pixel con etichetta 255 vengono ignorati automaticamente
        in_mask  = (ood_gts == 0)
        ood_mask = (ood_gts == 1)
        for m in methods:
            in_scores[m].append(scores[m][in_mask])
            ood_scores[m].append(scores[m][ood_mask])

        del S, L, crops
        if device == "cuda":
            torch.cuda.empty_cache()

    # Concatena tutti i pixel accumulati e calcola le metriche finali
    print(f"\n=== Risultati (T={args.temperature}) ===")
    for m in methods:
        val_out   = np.concatenate(in_scores[m]  + ood_scores[m])
        val_label = np.concatenate(
            [np.zeros(len(a)) for a in in_scores[m]] +
            [np.ones(len(a))  for a in ood_scores[m]]
        )
        # AuPRC: robusta allo sbilanciamento delle classi (pixel OOD << pixel inlier)
        auprc = average_precision_score(val_label, val_out) * 100.0
        # FPR@95: tasso di falsi positivi quando il 95% degli OOD è correttamente rilevato
        fpr   = fpr_at_95_tpr(val_out, val_label) * 100.0
        print(f"{m:11s} | AuPRC: {auprc:6.2f} | FPR@95: {fpr:6.2f}")


if __name__ == "__main__":
    main()
