
# <p align=center> Finding any Waldo with Zero-Shot Invariant and Efficient Visual Search</p>
![Python 3.8](https://img.shields.io/badge/python-3.8-g) ![pytorch 1.12.0](https://img.shields.io/badge/pytorch-1.12.0-blue.svg)

Authors: Mengmi Zhang, Jiashi Feng, Keng Teck Ma, Joo Hwee Lim, Qi Zhao, and Gabriel Kreiman

Our paper is published in Nature Communication. This is the official PyTorch codes for the paper. (Matlab codes [HERE](https://github.com/kreimanlab/VisualSearchZeroShot?tab=readme-ov-file)) 

Free access to our manuscript [HERE](https://www.nature.com/articles/s41467-018-06217-x), supplementary material [HERE](http://klab.tch.harvard.edu/publications/PDFs/gk7627_supplement.pdf)

## Abstract
Searching for a target object in a cluttered scene constitutes a fundamental challenge in daily vision. Visual search must be selective enough to discriminate the target from distractors, invariant to changes in the appearance of the target, efficient to avoid exhaustive exploration of the image, and must generalize to locate novel target objects with zero-shot training. Previous work on visual search has focused on searching for perfect matches of a target after extensive category-specific training. Here, we show for the first time that humans can efficiently and invariantly search for natural objects in complex scenes. To gain insight into the mechanisms that guide visual search, we propose a biologically inspired computational model that can locate targets without exhaustive sampling and which can generalize to novel objects. The model provides an approximation to the mechanisms integrating bottom-up and top-down signals during search in natural scenes.


| ![Stimuli](images/cropped_2_1.jpg) | [![Target](images/waldo.JPG)](sampleimg/waldo.JPG) | [![attentionmap](GIF/AM.gif)](GIF/AM.gif) | [![fixatedplace](GIF/FP.gif)](GIF/FP.gif) |
| :--------------------------: | :---------------------------------------------------: | :---------------------------------------: | :---------------------------------------: |
|           Stimuli            |                        Target                         |   Attention Map predicted by our model    |               Fixated Place               |


## Framework
The Invariant Visual Search Network (IVSN) model consists of a deep feed-forward network that mimics processing of features along ventral visual cortex, a way of temporarily storing information about the target tentatively associated with pre-frontal cortex, modulation of visual features in a top-down fashion to generate an attention map, and sequential selection of fixation locations.

<div align="center">
<img src="images/IVSN_framework.png" alt="Editor" width="500">
</div>

## Visual search in natural images

### COCO dataset
run IVSN_COCO.ipynb

### Natural dataset
run IVSN_natural.ipynb


## Citation
If you find our repo useful for your research, please cite us:
```
@article{zhang2018finding,
  title={Finding any Waldo with zero-shot invariant and efficient visual search},
  author={Zhang, Mengmi and Feng, Jiashi and Ma, Keng Teck and Lim, Joo Hwee and Zhao, Qi and Kreiman, Gabriel},
  journal={Nature communications},
  volume={9},
  number={1},
  pages={3730},
  year={2018},
  publisher={Nature Publishing Group UK London}
}
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.

