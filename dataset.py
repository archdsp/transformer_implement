from datasets import load_dataset
from datasets.table import Table

ds = load_dataset("wmt/wmt14", "de-en")
trainTable: Table = ds.data["train"]

train = trainTable.to_pylist()
for train_data in train:
    print(train_data["translation"]["en"])
    print(train_data["translation"]["de"])
    break