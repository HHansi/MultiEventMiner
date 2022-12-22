# MultiEventMiner

MultiEventMiner supports transformer-based cross-lingual news event detection at different levels of data granularity 
(i.e. sentence level and token level), involving different language-based learning strategies and a novel two-phase transfer 
learning strategy.

More details will be available with the paper "TTL: Transformer-based Two-phase Transfer Learning for Cross-lingual News 
Event Detection" which is under the review process currently.

## Installation
PyTorch needs to be installed first (preferably inside a conda environment). Please refer to 
[PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command 
for your platform, and refer to the [requirements.txt](https://github.com/HHansi/MultiEventMiner/blob/master/requirements.txt) for the required version details.

Then, the remaining libraries listed in [requirements.txt](https://github.com/HHansi/MultiEventMiner/blob/master/requirements.txt) also need to be installed.

## Experiments
### Train models
[train.py](https://github.com/HHansi/MultiEventMiner/blob/master/experiments/train.py) contains all the functions developed to train sentence and token level models.

### Predict
[predict.py](https://github.com/HHansi/MultiEventMiner/blob/master/experiments/predict.py) contains all the functions developed to make predictions at sentence and token levels using trained models. 


