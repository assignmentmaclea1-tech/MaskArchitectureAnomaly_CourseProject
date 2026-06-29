# Code for evaluating IoU 
# Nov 2017
# Eduardo Romera
#######################

# Questo modulo fornisce tre utility per la valutazione della segmentazione semantica:
# 1. iouEval: calcola IoU (Intersection over Union) per-classe e mean IoU (mIoU)
#    accumulando TP, FP, FN su più batch, con supporto per una classe ignore.
# 2. colors: codici per output colorato su terminale.
# 3. getColorEntry: mappa un valore float [0,1] a un colore in base alla qualità.

import torch

class iouEval:
    def __init__(self, nClasses, ignoreIndex=19):
        self.nClasses = nClasses
        # Se ignoreIndex >= nClasses non esiste come classe valida: si disabilita.
        # In Cityscapes la classe ignore è tipicamente l'indice 19 (su 19 classi 0-18),
        # quindi la condizione nClasses > ignoreIndex è vera e viene mantenuta.
        self.ignoreIndex = ignoreIndex if nClasses > ignoreIndex else -1
        self.reset()

    def reset(self):
        # Se c'è una classe ignore, i contatori hanno dimensione nClasses-1:
        # la classe ignore non contribuisce al calcolo dell'IoU.
        classes = self.nClasses if self.ignoreIndex == -1 else self.nClasses - 1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()

    def addBatch(self, x, y):  # x=predizioni, y=ground truth
        # Input atteso: (batch_size, nClasses, H, W) in formato one-hot,
        # oppure (batch_size, 1, H, W) con indici di classe da convertire.

        if x.is_cuda or y.is_cuda:
            x = x.cuda()
            y = y.cuda()

        # Conversione da indici di classe a one-hot se necessario.
        # scatter_(dim, index, value): per ogni posizione (b, 0, h, w) con valore c,
        # imposta x_onehot[b, c, h, w] = 1. Il risultato è un tensore binario
        # (batch, nClasses, H, W) dove esattamente un canale è 1 per ogni pixel.
        if x.size(1) == 1:
            x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3))
            if x.is_cuda:
                x_onehot = x_onehot.cuda()
            x_onehot.scatter_(1, x, 1).float()
        else:
            x_onehot = x.float()

        if y.size(1) == 1:
            y_onehot = torch.zeros(y.size(0), self.nClasses, y.size(2), y.size(3))
            if y.is_cuda:
                y_onehot = y_onehot.cuda()
            y_onehot.scatter_(1, y, 1).float()
        else:
            y_onehot = y.float()

        # Gestione della classe ignore: si estrae il canale corrispondente come
        # maschera binaria (1 dove il GT è ignore), poi si tronca sia x che y
        # rimuovendo quel canale. La maschera viene usata sotto per sottrarre
        # i pixel ignore dal conteggio dei FP, evitando di penalizzare predizioni
        # su pixel che non dovrebbero essere valutati.
        if self.ignoreIndex != -1:
            ignores = y_onehot[:, self.ignoreIndex].unsqueeze(1)  # (B, 1, H, W)
            x_onehot = x_onehot[:, :self.ignoreIndex]             # rimuove canale ignore
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores = 0  # scalare: non influenza i calcoli sotto

        # TP: pixel dove sia predizione che GT indicano la stessa classe.
        # Il prodotto elemento per elemento è 1 solo dove entrambi sono 1.
        tpmult = x_onehot * y_onehot
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()

        # FP: pixel dove la predizione dice "classe C" ma il GT dice "non C".
        # Si sottrae ignores per escludere i pixel ignore dal conteggio:
        # (1 - y_onehot - ignores) è 1 solo se GT=0 E ignore=0.
        # Senza questa sottrazione, predire una classe su un pixel ignore
        # verrebbe contato come FP, distorcendo l'IoU verso il basso.
        fpmult = x_onehot * (1 - y_onehot - ignores)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()

        # FN: pixel dove il GT dice "classe C" ma la predizione dice "non C".
        # I pixel ignore non compaiono qui perché y_onehot è già stato troncato
        # prima di questa operazione: il canale ignore è stato rimosso.
        fnmult = (1 - x_onehot) * y_onehot
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()

        # Accumulo su CPU in double per evitare overflow con dataset grandi:
        # con immagini 512x1024 e molti batch, i contatori possono superare 2^24
        # (limite di precisione di float32).
        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        # IoU per classe = TP / (TP + FP + FN), formula standard di Jaccard.
        # +1e-15 evita divisione per zero per classi assenti nel dataset.
        # Restituisce sia il mIoU (media su tutte le classi) sia il vettore
        # per-classe, utile per analizzare quali classi il modello gestisce peggio.
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return torch.mean(iou), iou  # (mIoU scalare, IoU per classe)


# Codici per colorare l'output su terminale.
# Il formato è '\033[<stile>;<colore>m' dove 31-36 sono i colori e 1 = bold.
class colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'  # reset: riporta il terminale al colore di default

# Mappa un valore IoU float [0,1] a un colore per feedback visivo immediato
# sulla qualità della segmentazione: rosso per valori scarsi, verde per ottimi.
def getColorEntry(val):
    if not isinstance(val, float):
        return colors.ENDC
    if val < .20:
        return colors.RED       # IoU molto basso: classe quasi mai predetta correttamente
    elif val < .40:
        return colors.YELLOW    # IoU mediocre
    elif val < .60:
        return colors.BLUE      # IoU accettabile
    elif val < .80:
        return colors.CYAN      # IoU buono
    else:
        return colors.GREEN     # IoU ottimo (>80%)

