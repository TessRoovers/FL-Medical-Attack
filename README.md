# Adversarial Attacks on Medical Image Segmentation in Federated Learning

## Overview
This repository contains the code and documentation for my bachelor thesis on adversarial attacks on medical image segmentation in federated learning.
More information can be found in the file *thesis.pdf*.

## Abstract
In this study, the impact of adversarial attacks on medical image segmentation in federated learning was researched. 
Experiments were conducted through training a variety of U-Net segmentation models, using the same chest x-ray dataset. 
Using different parameter values to create perturbations in one out of three clients’ training data, global models appear to be robust against these attacks.
Regularisation effects occurred for almost all model configurations, with only Grad-CAM Gaussian noise indicating adversarial attacks negatively impacting the global model performance. 
Future research into targeted attacks is required to gain a better understanding of adversarial attacks in federated learning.

## Installation and Usage
### Prerequisites
- Python 3.10.x
- pip


### Installation
Clone the repository and install the necessary packages using 'pip':
```bash
git clone https://github.com/TessRoovers/FL-Medical-Attack.git
cd FL-Medical-Attack
pip install -r requirements.txt
```

### Usage
Run the main script using the following parameters:
```bash
python3 ./train_gc.py --probability --std --label
```

Refer to *job_all.sh* for training multiple parameter values.

## Credits
```markdown
This project was built using the U-Net model architecture from the following repository:

- [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) licensed under the GNU General Public License v3.0
```

