# EoMT

This is almost the original repository of the authors of EoMT if something is not clear refer to the [original repo](https://github.com/tue-mps/eomt). For the given project tasks, the repository was modified. You will have to use the code in this folder and adapt it with the eval folder to be able to evaluate and train a EoMT model if needed. You can find a EoMT model trained on Cityscapes dataset with the [config file](eomt/configs/dinov2/cityscapes/semantic) at this [link](https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav?usp=drive_link).

## Requirements Installation
All the requirements installation are provided in the requirements.txt file. To install it, set the current directory appropriately and then use the command:
'''
!pip install -r requirements.txt
'''
## EVALUATING PART
## evalAnomaly_eomt.py

This code adapt evalAnomaly.py in the \eval\ folder to use EoMT pretrained model. The weights for this pretrained model can be downloaded [here](https://drive.google.com/file/d/1Xrglbc8y2izbQjUDmJASTegNWKAotXNp/view?usp=sharing)

**Examples of Inference Command:**
  ```
  python evalAnomaly_eomt.py \
  --input "/content/drive/MyDrive/Anomaly_Validation_Datasets/Validation_Dataset/RoadAnomaly21/images/*.png" \
  --ckpt_path "/content/drive/MyDrive/MaskArchitectureAnomaly_CourseProject-main/trained_models/eomt_cityscapes.bin" \
  --config "/content/MaskArchitectureAnomaly_CourseProject/eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml" \
  --post_hoc "MSP"
  ```
* Change the paths accordingly. The post-methods available are: "MSP", "MaxLogit", "MaxEntropy", "RbA".

## eval_iou.py

This code calculates the mIoU for the EoMT model on the CityScapes dataset with respect of different values of temperature.

**Examples of Inference Command:**
  ```
  !python eval_iou.py \
  --input "/content/cityscapes/leftImg8bit/val" \
  --gt_dir "/content/cityscapes/gtFine/val" \
  --ckpt_path "/content/drive/MyDrive/eomt_cityscapes.bin" \
  --config "/content/MaskArchitectureAnomaly_CourseProject/eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml" \
  --temperature 0.75

  ```
* The values for temperature are 0.5, 0.75, 1.0, 1.1

## Project Extension Part
In this part, file mask_classification_loss was updated to deal with the Logit Normalization Loss (the original file is still in the folder \training\ under the name mask_classification_loss_ORIGINAL.py). For evaluating, access the right directory and run the following command:
'''
!python main.py fit \
--config configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
--data.path /content \
--data.init_args.num_workers 0 \
--data.init_args.batch_size 1 \
--ckpt_path (if needed)
'''
* Change the paths accordingly.
