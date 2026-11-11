import sys
sys.path.append('../')

from public_module.data2 import read_csv_generator
from paddlenlp.datasets import load_dataset

train_set_path = '../literature_dataset/recall/train.csv'
train_ds = load_dataset(read_csv_generator, data_path=train_set_path, lazy=False)

# 展示：读取的数据类型
print(type(train_ds))

# 展示：前三行数据
for i in range(3):
    print(train_ds[i])