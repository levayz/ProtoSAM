# ProtoSAM - One shot segmentation with foundational models

Link to our paper [here](https://arxiv.org/abs/2407.07042). \
This work is the successor of [DINOv2-based-Self-Supervised-Learning](https://github.com/levayz/DINOv2-based-Self-Supervised-Learning) (Link to [Paper](arxiv.org/abs/2403.03273)).

## Abstract
This work introduces a new framework, ProtoSAM, for one-shot image segmentation. It combines DINOv2, a vision transformer that extracts features from images, with an Adaptive Local Prototype Pooling (ALP) layer, which generates prototypes from a support image and its mask. These prototypes are used to create an initial coarse segmentation mask by comparing the query image's features with the prototypes.
Following the extraction of an initial mask, we use numerical methods to generate prompts, such as points and bounding boxes, which are then input into the Segment Anything Model (SAM), a prompt-based segmentation model trained on natural images. This allows segmenting new classes automatically and effectively without the need for additional training. 

## How To Run
### 1. Data preprocessing
#### 1.1 CT and MRI Dataset
Please see the notebook `data/data_processing.ipynb` for instructions.
For convenience i've compiled the data processing instructions from https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation to a single notebook.  \
The CT dataset is available here: https://www.synapse.org/Synapse:syn3553734 \
The MRI dataset is availabel here: https://chaos.grand-challenge.org

run `./data/CHAOST2/dcm_img_to_nii.sh` to convert dicom images to nifti files.

#### 1.2 Polyp Dataset
Data is available here: https://www.kaggle.com/datasets/hngphmv/polypdataset?select=train.csv

Put the dataset `data/PolypDataset/`

### 2. Running
#### 2.1 (Optional) Training and Validation of the coarse segmentation networks
```
./backbone.sh [MODE] [MODALITY] [LABEL_SET]
```
MODE - validation or training \
MODALITY - ct or mri \
LABEL_SET - 0 (kidneys), 1 (liver spleen)

for example:
```
./backbone.sh training mri 1
```
Please refer to `backbone.sh` for further configurations.

#### 2.1 Running ProtoSAM
Put all SAM checkpoint like sam_vit_b.pth, sam_vit_h.pth, medsam_vit_b.pth into the `pretrained_model` directory. \
Checkpoints are available at [SAM](https://github.com/facebookresearch/segment-anything) and [MedSAM](https://github.com/bowang-lab/MedSAM).

```
./run_protosam.sh [MODALITY] [LABEL_SET]
```
MODALITY - ct, mri or polyp \
LABEL_SET (only relevant if doing ct or mri) - 0 (kidneys), 1 (liver spleen) 
Please refer to the `run_protosam.sh` script for further configurations.


## Acknowledgements
This work is largely based on [ALPNet](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation), [DINOv2](https://github.com/facebookresearch/dinov2), [SAM](https://github.com/facebookresearch/segment-anything) and is a continuation of [DINOv2-based-Self-Supervised-Learning](https://github.com/levayz/DINOv2-based-Self-Supervised-Learning).

## Cite
If you found this repo useful, please consider giving us a citation and a star!

```bibtex
@article{ayzenberg2024protosam,
  title={ProtoSAM-One Shot Medical Image Segmentation With Foundational Models},
  author={Ayzenberg, Lev and Giryes, Raja and Greenspan, Hayit},
  journal={arXiv preprint arXiv:2407.07042},
  year={2024}
}

@misc{ayzenberg2024dinov2,
      title={DINOv2 based Self Supervised Learning For Few Shot Medical Image Segmentation}, 
      author={Lev Ayzenberg and Raja Giryes and Hayit Greenspan},
      year={2024},
      eprint={2403.03273},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
