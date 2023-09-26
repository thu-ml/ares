<div align="center">

# 🚀 Welcome to **ARES 2.0** 🚀

</div>

## 🌐 Overview

🔍 **ARES 2.0** (Adversarial Robustness Evaluation for Safety) is a Python library dedicated to adversarial machine learning research. It aims at benchmarking the adversarial robustness of image classification and object detection models, and introduces mechanisms to defend against adversarial attacks through robust training.


## 🌟 Features

-  Developed on **Pytorch**.
- Supports [various attacks](/ares/attack/__init__.py) on classification models.
- Employs adversarial attacks on object detection models.
- Provides robust training for enhanced robustness and various trained **checkpoints**.
- Enables distributed training and testing.


## 💾 Installation

1. **Optional**: Initialize a dedicated environment for ARES 2.0.
   
   ```
   conda create -n ares python==3.10.9
   conda activate ares
   ```
2. Clone and set up ARES 2.0 via the following commands:
   
   ```
   git clone https://github.com/thu-ml/ares2.0
   cd ares2.0
   pip install -r requirements.txt
   mim install mmengine==0.8.4
   mim install mmcv==2.0.0 
   mim install mmdet==3.1.0
   pip install -v -e .
   ```

## 🚀 Getting Started

- For robustness evaluation of image classification models against adversarial attacks, please refer to [classification](./classification/README.md).
- For robustness evaluation of object detection models, please refer to [detection](./detection/README.md).
- For methodologies on robust training, please refer to [robust-training](./robust_training/README.md).


## 📘 Documentation

📚 Access detailed **tutorials** and **API docs** on strategies to attack classification models, object detection models, and robust training [here](https://thu-ml.github.io/ares/).


## 📝 Citation

If you derive value from ARES 2.0 in your endeavors, kindly cite our  paper on adversarial robustness, which encompasses all models, attacks, and defenses incorporated in ARES 2.0:

```
@inproceedings{dong2020benchmarking,
  title={Benchmarking Adversarial Robustness on Image Classification},
  author={Dong, Yinpeng and Fu, Qi-An and Yang, Xiao and Pang, Tianyu and Su, Hang and Xiao, Zihao and Zhu, Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={321--331},
  year={2020}
}
```
