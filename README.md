# Generative Adversarial Ranking Nets

![GARNet](https://github.com/EvaFlower/GARNet/blob/main/GARNet_framework.png?raw=true)

Code for [Generative Adversarial Ranking Nets](https://jmlr.org/papers/v25/23-0461.html) (JMLR 2024). 

## Environment
After installing [Anaconda](https://www.anaconda.com/), create a new Conda environment using `conda env create -f environment.yml`.

`pip install allrank --no-deps`

## Dataset
- MNIST
- **[LFW](https://vis-www.cs.umass.edu/lfw/)**
- **[UT-Zap50K](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/)**  

Download datasets and update the data directory in read_*_dataset.py

## Pretrained ranker
Download pretrained ranker at https://drive.google.com/drive/folders/123GD7vO_ZMtcXtMQV2Kj5PK_0UCzFwlE?usp=share_link  
Put pretrained_ranker at the directory UT-Zap50K/.

## Training
`python adversarial_ranking.py`

## Evaluation
`python eval_ranking.py`  
`python eval_fid.py`
