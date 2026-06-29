# EOMT
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

This code calculates the predicted mIoU for the EoMT model on the CityScapes dataset with respect of different values of temperature.

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
