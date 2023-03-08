# MultiEventMiner

MultiEventMiner supports transformer-based cross-lingual news event detection at different levels of data granularity 
(i.e. sentence level and token level), involving different language-based learning strategies and a novel two-phase transfer 
learning strategy.

For more details, please refer to our paper ["TTL: transformer-based two-phase transfer learning for cross-lingual news event detection"](https://link.springer.com/article/10.1007/s13042-023-01795-9)

If you use this system, please consider citing this paper; reference details are given below.

## Installation
PyTorch needs to be installed first (preferably inside a conda environment). Please refer to 
[PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command 
for your platform, and refer to the [requirements.txt](https://github.com/HHansi/MultiEventMiner/blob/master/requirements.txt) for the required version details.

Then, the remaining libraries listed in [requirements.txt](https://github.com/HHansi/MultiEventMiner/blob/master/requirements.txt) also need to be installed.

## Experiments
### Train models
Refer to [train.py](https://github.com/HHansi/MultiEventMiner/blob/master/experiments/train.py) to get access to all the functions developed to train sentence and token level models.

### Predict
Refer to [predict.py](https://github.com/HHansi/MultiEventMiner/blob/master/experiments/predict.py) to get access to all the functions developed to make predictions at sentence and token levels using trained models. 

## Reference
```
@article{hettiarachchi2023ttl,
title = {{TTL}: transformer-based two-phase transfer learning for cross-lingual news event detection},
author = {Hettiarachchi, Hansi and Adedoyin-Olowe, Mariam and Bhogal, Jagdev and Gaber, Mohamed Medhat},
journal = {International Journal of Machine Learning and Cybernetics},
year = {2023},
publisher={Springer},
doi = {10.1007/s13042-023-01795-9},
url = {https://doi.org/10.1007/s13042-023-01795-9}
}
```

