# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from evalAonomaly_eomt import build_eomt_model
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = build_eomt_model(config_path, ckpt_path, device)

    #model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")


    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset), num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    iouEvalVal = iouEval(NUM_CLASSES)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1] 

        print (step, filenameSave)


    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--ckpt_path', default="eomt_cityscapes.bin")
    parser.add_argument('--loadModel', default="evalAnomaly_eomt.py")
    parser.add_argument('--config_path', default="eomt_base_640.yaml")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--input', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
'''
# ---------------------------------------------------------------
# eval_iou.py - Valutazione mIoU di EoMT su Cityscapes (val)
#               con Temperature Scaling a diverse T.
#
# NOTA CONCETTUALE (importante per la relazione):
#   Il mIoU si calcola dall'ARGMAX dei logit per-pixel. Dividere i
#   logit per una costante T > 0 non cambia l'argmax di una softmax
#   pixel-wise classica  ->  in quel caso il mIoU sarebbe IDENTICO per
#   ogni T. In EoMT la temperatura entra DENTRO il softmax sulle query
#   ( softmax(class / T) ) e poi c'e' una combinazione pesata con le
#   maschere: quindi l'argmax (e di conseguenza il mIoU) PUO' variare,
#   ma di pochissimo. La temperature scaling serve soprattutto alla
#   CALIBRAZIONE / anomaly detection, non all'accuratezza di segmentazione.
#
# Esempio d'uso (Colab), lanciato dalla cartella eomt/:
#   python eval_iou.py \
#       --datadir   /content/cityscapes \
#       --ckpt_path /content/MaskArchitectureAnomaly_CourseProject/trained_models/eomt_cityscapes.bin \
#       --temperatures 0.5 1.0 1.5 2.0
#
# Per una run piu' veloce/leggera:  --height 512 --width 1024
# ---------------------------------------------------------------

import os
import sys
import math
import time
import yaml
import importlib
import warnings
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# La cartella che contiene questo script (eomt/) deve stare nel sys.path
# per poter importare dinamicamente models.*, training.*, datasets.*
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from iouEval import iouEval, getColorEntry  # sta nella stessa cartella eomt/


# ============================================================
# 1. COSTRUZIONE DEL MODELLO EoMT (da YAML + checkpoint locale)
# ============================================================
def build_eomt_model(config_path, ckpt_path, device):
    """Costruisce EoMT dal config YAML e carica i pesi dal checkpoint .bin locale."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    warnings.filterwarnings("ignore", message=r".*is already saved during checkpointing.*")

    img_size = config.get("data", {}).get("init_args", {}).get("img_size", (640, 640))
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    img_size = tuple(img_size)
    num_classes = 19  # Cityscapes

    # --- encoder (es. ViT/DINOv2) ---
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_cls = encoder_cfg["class_path"].rsplit(".", 1)
    encoder = getattr(importlib.import_module(enc_mod), enc_cls)(
        img_size=img_size, **encoder_cfg.get("init_args", {})
    )

    # --- rete EoMT completa ---
    net_cfg = config["model"]["init_args"]["network"]
    net_mod, net_cls = net_cfg["class_path"].rsplit(".", 1)
    net_kwargs = {k: v for k, v in net_cfg["init_args"].items() if k != "encoder"}
    network = getattr(importlib.import_module(net_mod), net_cls)(
        masked_attn_enabled=False,            # la masked attention serve solo in training
        num_classes=num_classes,
        encoder=encoder,
        **net_kwargs,
    )

    # --- Lightning module che wrappa la rete ---
    lit_mod, lit_cls = config["model"]["class_path"].rsplit(".", 1)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    model = getattr(importlib.import_module(lit_mod), lit_cls)(
        img_size=img_size,
        num_classes=num_classes,
        network=network,
        **model_kwargs,
    )

    # --- checkpoint locale (.bin = state_dict grezzo) ---
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]  # tollera eventuali .ckpt di Lightning

    # Adatta il positional embedding se la risoluzione del checkpoint differisce
    pos_key = "network.encoder.backbone.pos_embed"
    if pos_key in state_dict:
        ckpt_pos = state_dict[pos_key]
        model_pos = model.state_dict()[pos_key]
        if ckpt_pos.shape != model_pos.shape:
            C = ckpt_pos.shape[-1]
            g_ckpt = int(math.sqrt(ckpt_pos.shape[1]))
            g_model = int(math.sqrt(model_pos.shape[1]))
            pos = ckpt_pos.reshape(1, g_ckpt, g_ckpt, C).permute(0, 3, 1, 2)
            pos = F.interpolate(pos, size=(g_model, g_model), mode="bicubic", align_corners=False)
            state_dict[pos_key] = pos.permute(0, 2, 3, 1).reshape(1, g_model * g_model, C)

    missing = model.load_state_dict(state_dict, strict=False)
    print(f"Pesi caricati da {ckpt_path} | missing={len(missing.missing_keys)} "
          f"unexpected={len(missing.unexpected_keys)}")
    return model.to(device).eval()


# ============================================================
# 2. DATASET: accoppia immagini e GT (labelTrainIds) di Cityscapes
# ============================================================
def list_cityscapes(datadir, subset="val"):
    """Ritorna due liste ordinate e allineate: percorsi immagini e percorsi GT."""
    img_root = os.path.join(datadir, "leftImg8bit", subset)
    gt_root = os.path.join(datadir, "gtFine", subset)

    images = sorted(
        os.path.join(dp, f)
        for dp, _, fns in os.walk(os.path.expanduser(img_root))
        for f in fns if f.endswith(".png")
    )
    labels = sorted(
        os.path.join(dp, f)
        for dp, _, fns in os.walk(os.path.expanduser(gt_root))
        for f in fns if f.endswith("_labelTrainIds.png")
    )
    return images, labels


# ============================================================
# 3. MAIN
# ============================================================
NUM_CLASSES = 20    # 19 classi valide (0..18) + 1 indice di ignore (19)
IGNORE_INDEX = 19

CLASS_NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
               "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
               "truck", "bus", "train", "motorcycle", "bicycle"]


def main():
    parser = ArgumentParser()
    parser.add_argument("--datadir", required=True,
                        help="root Cityscapes (contiene leftImg8bit/ e gtFine/)")
    parser.add_argument("--ckpt_path",
                        default=os.path.join(SCRIPT_DIR, "..", "trained_models", "eomt_cityscapes.bin"))
    parser.add_argument("--config",
                        default=os.path.join(SCRIPT_DIR, "configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"))
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0],
                        help="lista di T, es. --temperatures 0.5 1.0 1.5 2.0")
    parser.add_argument("--subset", default="val")
    parser.add_argument("--height", type=int, default=1024, help="risoluzione di valutazione (H)")
    parser.add_argument("--width", type=int, default=2048, help="risoluzione di valutazione (W)")
    parser.add_argument("--max_images", type=int, default=None, help="limita il n. di immagini (debug)")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    H, W = args.height, args.width
    print(f"Device: {device} | risoluzione di eval: {H}x{W} | T = {args.temperatures}\n")

    print(f"Carico il modello (config={os.path.basename(args.config)})...")
    model = build_eomt_model(args.config, args.ckpt_path, device)
    print("Modello pronto.\n")

    images, labels = list_cityscapes(args.datadir, args.subset)
    assert len(images) > 0, f"Nessuna immagine in {args.datadir}/leftImg8bit/{args.subset}"
    assert len(images) == len(labels), (
        f"Immagini ({len(images)}) e label *_labelTrainIds.png ({len(labels)}) non combaciano. "
        "Hai generato le labelTrainIds (createTrainIdLabelImgs.py di cityscapesscripts)?"
    )
    if args.max_images:
        images, labels = images[:args.max_images], labels[:args.max_images]
    print(f"{len(images)} immagini da valutare.\n")

    # Un valutatore IoU separato per ogni temperatura
    evaluators = {T: iouEval(NUM_CLASSES) for T in args.temperatures}

    start = time.time()
    for step, (img_path, gt_path) in enumerate(zip(images, labels)):
        # --- immagine: uint8 [3,H,W], formato atteso da EoMT ---
        pil = Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR)
        img = torch.from_numpy(np.array(pil)).permute(2, 0, 1).contiguous().to(device)
        imgs = [img]
        img_sizes = [img.shape[-2:]]

        # --- label: [1,1,H,W], 255 (ignore) -> 19 ---
        gt = Image.open(gt_path).resize((W, H), Image.NEAREST)
        gt = torch.from_numpy(np.array(gt)).long()
        gt[gt == 255] = IGNORE_INDEX
        gt = gt.view(1, 1, H, W).to(device)

        # --- FORWARD PASS una sola volta (e' indipendente da T) ---
        with torch.no_grad():
            crops, origins = model.window_imgs_semantic(imgs)
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    mask_layers, class_layers = model(crops)
            else:
                mask_layers, class_layers = model(crops)

            # Solo l'ultimo layer del decoder; interpola le maschere a img_size (640x640)
            mask_logits = F.interpolate(
                mask_layers[-1].float(), model.img_size, mode="bilinear", align_corners=False
            )                                       # [N, Q, 640, 640]
            class_logits = class_layers[-1].float()  # [N, Q, C+1]

            # --- ciclo SOLO sulle T (parte economica: niente nuovo forward) ---
            for T in args.temperatures:
                # Temperatura DENTRO il softmax sulle classi; si scarta la no-object (-1)
                cls_soft = (class_logits / T).softmax(dim=-1)[..., :-1]          # [N, Q, C]
                crop_logits = torch.einsum(
                    "nqhw,nqc->nchw", mask_logits.sigmoid(), cls_soft
                )                                                                # [N, C, h, w]
                # Riassembla i crop alla risoluzione originale dell'immagine
                logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)[0]
                pred = logits.argmax(0).view(1, 1, H, W)                         # [1,1,H,W]
                evaluators[T].addBatch(pred, gt)

        if step % 25 == 0:
            print(f"  [{step + 1}/{len(images)}] {os.path.basename(img_path)}")

        del crops, mask_layers, class_layers, mask_logits, class_logits
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nValutazione completata in {time.time() - start:.1f}s")

    # --- risultati dettagliati per ogni T ---
    summary = {}
    for T in args.temperatures:
        iou_mean, iou_classes = evaluators[T].getIoU()
        summary[T] = iou_mean.item() * 100
        print("\n" + "=" * 55)
        print(f" RISULTATI   T = {T}")
        print("=" * 55)
        for i, name in enumerate(CLASS_NAMES):
            v = iou_classes[i].item()
            print(f"  {getColorEntry(v)}{v * 100:6.2f}\033[0m  {name}")
        print("-" * 55)
        m = iou_mean.item()
        print(f"  {getColorEntry(m)}mIoU = {m * 100:.2f}%\033[0m")

    # --- tabella riassuntiva mIoU vs Temperatura ---
    print("\n" + "=" * 32)
    print("  mIoU  vs  Temperature")
    print("=" * 32)
    for T in args.temperatures:
        print(f"  T = {T:<5}  ->  mIoU = {summary[T]:.2f}%")
    print("=" * 32)


if __name__ == "__main__":
    main()
'''
