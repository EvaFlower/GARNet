# Generative Adversarial Ranking Nets

Code for [Generative Adversarial Ranking Nets]() (JMLR 2024). 

## Environment
After installing [Anaconda](https://www.anaconda.com/), create a new Conda environment using `conda env create -f environment.yml`.

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
