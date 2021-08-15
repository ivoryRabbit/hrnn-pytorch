# HRNN-Pytorch
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Paper
[Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks](https://arxiv.org/pdf/1706.04148.pdf)

## Dataset
[ml-1m](https://grouplens.org/datasets/movielens/1m/)

## Clone
```bash
git clone git@github.com:ivoryRabbit/hrnn-pytorch.git
```

## Requirements
```bash
pip install -r requirements.txt
```

## Preprocess
```bash
python preprocess.py --interactioin ml-1m.csv
```

## Training
```bash
python train.py --n_epochs 10 --batch_size 50
```

## Recommend
```bash
python main.py --user_id 2 --eval_k 10
```

## Reference
- [original code for the paper with Theano](https://github.com/mquad/hgru4rec)