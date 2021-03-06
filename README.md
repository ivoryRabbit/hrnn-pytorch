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

## Load Data
```bash
python3 load_data.py
```

## Preprocess
```bash
python3 preprocess.py --interaction ml-1m.csv
```

## Training
```bash
python3 train.py --n_epochs 10 --batch_size 50
```

## Recommend
```bash
python3 main.py --user_id 2 --eval_k 10
```

## Reference
- [HGRU4REC-Theano](https://github.com/mquad/hgru4rec)
- [GRU4REC-PyTorch](https://github.com/hungthanhpham94/GRU4REC-pytorch)