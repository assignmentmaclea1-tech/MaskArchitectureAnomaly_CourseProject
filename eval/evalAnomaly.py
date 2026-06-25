#This script is based on the original evalAnomaly.py (now evalAnomaly_original.py in this folder). The key change here is the 
#introduction of Max Logit and Max Entropy as post-hoc methods.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser

from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn.functional as F

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--post_hoc', default="MSP", choices=["MSP", "MaxLogit", "MaxEntropy"], help="Scegli la metrica da usare")
    args = parser.parse_args()

    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)
    print(f"Using Post-Hoc Method: {args.post_hoc}")

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):
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

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(f"Processing: {os.path.basename(path)}")

        # QUI ERA L'ERRORE: Ho tolto il .permute() che sfalsava le dimensioni!
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()

        with torch.no_grad():
            result = model(images)

        logits = result.squeeze(0)
        probs = F.softmax(logits, dim=0)

        if args.post_hoc == "MSP":
            anomaly_result = 1.0 - probs.max(dim=0)[0].cpu().numpy()
        elif args.post_hoc == "MaxLogit":
            anomaly_result = -logits.max(dim=0)[0].cpu().numpy()
        elif args.post_hoc == "MaxEntropy":
            anomaly_result = -torch.sum(probs * torch.log(probs + 1e-8), dim=0).cpu().numpy()

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

        del result, logits, probs, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")

    if len(ood_gts_list) == 0:
        print("Nessuna immagine valida trovata.")
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
