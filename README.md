# HRNN-Pytorch
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Paper
[Personalizing Session-based Recommendations with Hierarchical Recurrent Neural Networks](https://arxiv.org/pdf/1706.04148.pdf)

## Dataset
[ml-1m](https://grouplens.org/datasets/movielens/1m/)

## Requirements
```bash
pip install -r requirements.txt
```

## Preprocess
```bash
python preprocess.py --raw_data ml-1m.csv
```

## Training
```bash
python train.py --n_epochs 10 --batch_size 50
```

## Inference
```bash
python recommend.py --user_id 32
```