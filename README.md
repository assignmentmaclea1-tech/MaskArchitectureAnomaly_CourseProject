# Mask Architecture for Road Scenes
This is the repository for [Mask Architecture Anomaly Segmentation for Road Scenes](https://drive.google.com/file/d/1Vz08DHsP_mojpCTAQTR6NHVq-2rEqAZM/view).\
This repository consists of the code base for training/testing ERFNet on the Cityscapes dataset and perform anomaly segmentation. It also contains some code referring to EoMT. Some of this code may be unnecessary.

## Folders
For instructions, please refer to the README in each folder:

* [eval](eval) contains tools for evaluating/visualizing an ERFNet model's output and performing anomaly segmentation.
* [trained_models](trained_models) Contains the ERFNet trained models for the baseline eval. 
* [eomt](eomt) It is almost the original folder of the EoMT project. Inside it you will find code to train and pretrained checkpoints for EoMT.

## TO DO
* Ho rinominato il file originale di mask_classification_loss per creare un file .py con lo stesso nome con la loss necessaria per la Logit Norm. L'ho fatto per questioni di directory, non vorrei che si creassero casini.
* Ci sono file inutili. Sono da eliminare?
* Il README di eval è stato modificato. Il resto è pressochè identico.
* Ho creato appositi file _eval.py utili per punto 4,5 senza sovrascivere originali (anzi, quello originale è stato rinominato)
* In trained_models non sono presenti i pesi per EoMT "eomt_cityscapes.bin", ho lasciato un link nel README.md di eval per poterli scaricare.

