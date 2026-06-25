#COMMENT
#In the original Colab, evalAnomaly.py was overwritten, here I created a new file .py for the Temperature Scaling Task.
#The results are summarized in results.txt
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

# Import temperature_scaling
from temperature_scaling import ModelWithTemperature

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])

def main():
    parser = ArgumentParser()
    parser.add_argument("--input", default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp", nargs="+")
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--post_hoc', default="MSP", choices=["MSP", "MaxLogit", "MaxEntropy"])

    # ECCO IL PARAMETRO MANCANTE: Ora accetterà --temperature
    parser.add_argument('--temperature', type=float, default=1.0, help="Forza la temperatura (es. 0.5)")
    args = parser.parse_args()

    anomaly_score_list = []
    ood_gts_list = []

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print(f"Using Method: {args.post_hoc} | FORCED TEMPERATURE: {args.temperature}")

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
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))

    # Avvolgiamo il modello
    model = ModelWithTemperature(model)

    # Forziamo la temperatura al valore che passi dal comando
    model.temperature.data = torch.tensor([args.temperature]).cuda()

    print("Model, Weights and GitHub Temperature Scaling LOADED successfully")
    model.eval()

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):

        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()

        with torch.no_grad():
            logits = model(images)

        logits = logits.squeeze(0)
        probs = F.softmax(logits, dim=0)

        if args.post_hoc == "MSP":
            anomaly_result = 1.0 - probs.max(dim=0)[0].cpu().numpy()
        elif args.post_hoc == "MaxLogit":
            anomaly_result = -logits.max(dim=0)[0].cpu().numpy()
        elif args.post_hoc == "MaxEntropy":
            anomaly_result = -torch.sum(probs * torch.log(probs + 1e-8), dim=0).cpu().numpy()

        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT: pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT: pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT: pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT: ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
             ood_gts = np.where((ood_gts==0), 255, ood_gts)
             ood_gts = np.where((ood_gts==1), 0, ood_gts)
             ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
        if "Streethazard" in pathGT:
             ood_gts = np.where((ood_gts==14), 255, ood_gts)
             ood_gts = np.where((ood_gts<20), 0, ood_gts)
             ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts): continue
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)

        del logits, probs, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    if len(ood_gts_list) == 0:
        return

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    val_out = np.concatenate((anomaly_scores[ind_mask], anomaly_scores[ood_mask]))
    val_label = np.concatenate((np.zeros(len(anomaly_scores[ind_mask])), np.ones(len(anomaly_scores[ood_mask]))))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'Method: {args.post_hoc} | Temp: {args.temperature} | AUPRC: {prc_auc*100.0:.2f} | FPR@TPR95: {fpr*100.0:.2f}')

if __name__ == '__main__':
    main()
