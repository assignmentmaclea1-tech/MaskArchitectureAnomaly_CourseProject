# Questo modulo implementa una versione modificata della loss di Mask2Former che integra
# LogitNorm: una tecnica che normalizza i logits delle classi prima del calcolo della
# cross-entropy, con l'obiettivo di migliorare la calibrazione del modello e la
# separabilità tra classi in-distribution e OOD.
# La modifica principale è nel metodo loss_labels, dove i logits vengono normalizzati
# sulla sfera unitaria L2 e scalati da una temperatura tau prima di calcolare la loss.

from typing import List, Optional
import torch.distributed as dist
import torch
import torch.nn as nn
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
)

class MaskClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
    ):
        # Si inizializza direttamente nn.Module invece di chiamare super().__init__()
        # di Mask2FormerLoss: questo è intenzionale per evitare che il costruttore
        # padre sovrascriva parametri che qui vengono ridefiniti manualmente
        # (es. empty_weight, matcher). È una scelta fragile: se Mask2FormerLoss
        # aggiunge logica nel __init__ in futuro, qui verrebbe silenziosamente ignorata.
        nn.Module.__init__(self)

        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient

        # Il vettore empty_weight assegna peso 1.0 a tutte le classi reali e
        # no_object_coefficient all'ultima classe (indice num_labels), che rappresenta
        # la "non-classe" o background nel matching. Abbassare questo peso riduce
        # la penalità per query non matchate, bilanciando il training.
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # tau è il parametro centrale di LogitNorm: valori piccoli (es. 0.04)
        # producono logits normalizzati con norma ~1/tau = 25, forzando il modello
        # ad essere molto "deciso" durante il training. Questo è diverso dal
        # Temperature Scaling in inference (evalAnomaly_eomt_temp.py): qui tau
        # agisce sulla loss durante il training, non sulle probabilità a test time.
        self.tau = 0.04

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

    @torch.compiler.disable
    # Il decorator disabilita torch.compile su questo metodo: necessario perché
    # il Hungarian Matcher interno usa operazioni non tracciabili dal compilatore
    # (es. scipy linear_sum_assignment o loop Python dinamici).
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: Optional[torch.Tensor] = None,
    ):
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]

        # Il matcher risolve il problema di assegnazione ottimale tra le Q query
        # del modello e i G ground truth oggetti per ogni immagine nel batch.
        # Restituisce una lista di tuple (idx_query, idx_gt) per ogni immagine.
        # Questo matching è necessario perché Mask2Former produce un insieme non
        # ordinato di predizioni: non c'è corrispondenza a priori tra query i e GT j.
        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)

        return {**loss_masks, **loss_classes}

    def loss_labels(self, class_queries_logits, class_labels, indices):
        # Cuore di LogitNorm: normalizza ogni vettore di logits sulla sfera unitaria
        # e poi divide per tau. L'effetto geometrico è che tutti i vettori di logits
        # vengono proiettati sulla stessa ipersfera di raggio 1/tau, indipendentemente
        # dalla loro norma originale. Questo impedisce al modello di "barare" sulla
        # loss aumentando semplicemente la norma dei logits invece di imparare
        # feature discriminative. Il +1e-7 evita divisione per zero.
        norms = torch.norm(class_queries_logits, p=2, dim=-1, keepdim=True) + 1e-7
        normalized_logits = (class_queries_logits / norms) / self.tau

        # Print di debug: utile durante lo sviluppo per verificare che la norma
        # media post-normalizzazione sia circa 1/tau = 25. Da rimuovere in produzione
        # perché chiamato ad ogni forward pass, con impatto non trascurabile su I/O.
        print(f"[LOGITNORM ATTIVO] tau={self.tau}, norm_media={normalized_logits.norm(dim=-1).mean():.2f}")

        return super().loss_labels(normalized_logits, class_labels, indices)

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

        # Normalizzazione per numero di maschere totali nel batch: senza questo,
        # la loss crescerebbe linearmente con il numero di oggetti per immagine,
        # rendendo il gradiente instabile con batch di densità variabile.
        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        # In training distribuito (multi-GPU), ogni processo vede solo una parte
        # del batch. all_reduce somma num_masks tra tutti i processi e si divide
        # per world_size, ottenendo la media globale. Senza questo, la loss
        # sarebbe inconsistente tra GPU con numero diverso di oggetti nel proprio shard.
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)  # min=1 evita divisione per zero

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        # Mask2Former produce predizioni a più layer del decoder (auxiliary losses):
        # losses_all_layers contiene le loss di tutti i layer, non solo dell'ultimo.
        # Sommarle pesate stabilizza il training guidando anche i layer intermedi.
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            # I tre tipi di loss corrispondono ai tre termini di Mask2Former:
            # - mask: BCE sulla maschera binaria predetta vs GT
            # - dice: Dice loss sulla maschera (più robusta allo sbilanciamento)
            # - cross_entropy: classificazione della query matchata
            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total
