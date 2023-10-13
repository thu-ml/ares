# Attack-Bard

## Demos

---

![image](https://github.com/thu-ml/Attack-Bard/blob/main/dataset/demos/VQA.png)


## Introduction

---

Multimodal Large Language Models (MLLMs) that integrate text and other modalities (especially vision) have achieved unprecedented performance in various multimodal tasks. However, due to the unsolved adversarial robustness problem of vision models, MLLMs can have more severe safety and security risks by introducing the vision inputs. In this work, we study the adversarial robustness of Google's Bard, a competitive chatbot to ChatGPT that released its multimodal capability recently, to better understand the vulnerabilities of commercial MLLMs. 
By attacking white-box surrogate vision encoders or MLLMs, the generated adversarial examples can mislead Bard to output wrong image descriptions with a 22% success rate based solely on the transferability. We show that the adversarial examples can also attack other MLLMs, e.g., 26% attack success rate against Bing Chat and 86\% attack success rate against ERNIE bot. Moreover, we identify two defense mechanisms of Bard, including face detection and toxicity detection of images. We design corresponding attacks to evade these defenses, demonstrating that the current defenses of Bard are also vulnerable. We hope this work can deepen our understanding on the robustness of MLLMs and facilitate future research on defenses. 


We provide codes for 3 experiments.

1. attack_img_encoder_misdescription.py: Image embedding attack against Bard's image description. You can also use this code to attack NSFW detectors by changing the training data.

2. attack_vlm_misclassify.py: Text description attack against Bard's image description. 



## Getting Started

---

### Installation

The installation of this project is extremely easy. You only need to:

- Configurate the environment, vicuna weights, following the instruction in https://github.com/Vision-CAIR/MiniGPT-4    

and run the following codes

- Image embedding attack against Bard's image description. You can also use this code to attack NSFW detectors by changing the training data.
```
CUDA_VISIBLE_DEVICES=0,1,2 attack_img_encoder_misdescription.py
```
- Text description attack against Bard's image description. 
```
CUDA_VISIBLE_DEVICES=0 attack_vlm_misclassify.py
```


### Results

- Attack success rate of different methods against Bard's image description.


|                         | Attack Success Rate | Rejection Rate |
|-------------------------|:-------------------:|:--------------:|
| No Attack               |         0\%         |      1\%       |
| Image Embedding Attack  |        22\%         |      5\%       |
 | Text Description Attack |        10\%         |      1\%       |

- We achieve 36\% attack success rate against Bard's toxic detector.

  - Several demos:

![image](https://github.com/thu-ml/Attack-Bard/blob/main/dataset/demos/mis_1.jpg)

![image](https://github.com/thu-ml/Attack-Bard/blob/main/dataset/demos/mis_2.jpg)

![image](https://github.com/thu-ml/Attack-Bard/blob/main/dataset/demos/toxic_1.jpg)

![image](https://github.com/thu-ml/Attack-Bard/blob/main/dataset/demos/toxic_2.jpg)

![image](https://github.com/thu-ml/Attack-Bard/blob/main/dataset/demos/ffhq_1.jpg)

![image](https://github.com/thu-ml/Attack-Bard/blob/main/dataset/demos/ffhq_2.jpg)

![image](https://github.com/thu-ml/Attack-Bard/blob/main/dataset/demos/weixin1.jpg)

![image](https://github.com/thu-ml/Attack-Bard/blob/main/dataset/demos/weixin2.jpg)

![image](https://github.com/thu-ml/Attack-Bard/blob/main/dataset/demos/bing1.jpg)

![image](https://github.com/thu-ml/Attack-Bard/blob/main/dataset/demos/bing2.jpg)


# Acknowledgement

---

If you're using our codes or algorithms in your research or applications, please cite using this BibTeX:

```
@article{dong2023robust,
  title={How Robust is Google's Bard to Adversarial Image Attacks?},
  author={Dong, Yinpeng and Chen, Huanran and Chen, Jiawei and Fang, Zhengwei and Yang, Xiao and Zhang, Yichi and Tian, Yu and Su, Hang and Zhu, Jun},
  journal={arXiv preprint arXiv:2309.11751},
  year={2023}
}
```


Our code is implemented based on [**MiniGPT4**](https://github.com/Vision-CAIR/MiniGPT-4) and [**AdversarialAttacks**](https://github.com/huanranchen/AdversarialAttacks).  Thanks them for supporting! 



