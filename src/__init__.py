import os

os.environ["model_dir"] = "./trained/HRNN-TOP1.pt"

os.environ["train_dir"] = "./data/train_data.hdf"
os.environ["valid_dir"] = "./data/valid_data.hdf"
os.environ["test_dir"] = "./data/test_data.hdf"
os.environ["item_dir"] = "./data/item_for_train.csv"
