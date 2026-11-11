# %%
import os
from functools import partial

import numpy as np
import paddle
from public_module.data2 import convert_pairwise_example
from public_module.data2 import read_text_pair
from public_module.base_model import PairwiseMatching
import copy

from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModel, AutoTokenizer

@paddle.no_grad()
def predict(model, data_loader):
    model.eval()

    batch_probs = []
    for batch_data in data_loader:
        input_ids, token_type_ids = batch_data
        batch_prob = model.predict(input_ids=input_ids, token_type_ids=token_type_ids).numpy()
        batch_probs.append(batch_prob)

    conca_batch_probs = np.concatenate(batch_probs, axis=0)

    return conca_batch_probs
paddle.set_device("cpu")
#%% (1) 构建模型并载入模型参数
pretrained_model = AutoModel.from_pretrained("ernie-3.0-medium-zh")
model = PairwiseMatching(pretrained_model)

params_path = "model_param/model_400/model_state.pdparams"
if params_path and os.path.isfile(params_path): 
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)
else:
    raise ValueError("Please set --params_path with correct pretrained model file")

#%% (2) 加载数据，构建数据加载器
test_data_path = "../literature_dataset/rank/test_pairwise.csv"
test_ds = load_dataset(read_text_pair, data_path=test_data_path, lazy=False)
test_ds_copy = copy.deepcopy(test_ds)

tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")
trans_func = partial(convert_pairwise_example, tokenizer=tokenizer, max_seq_length=64, phase="predict")

batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=128, shuffle=False)
batchify_fn = lambda samples, fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64")
    ):[data for data in fn(samples)]
test_data_loader = paddle.io.DataLoader(dataset=test_ds.map(trans_func), batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)

#%% （3） 计算相似度
y_probs = predict(model, test_data_loader)

rank_result = []
for idx, prob in enumerate(y_probs):
    text_pair = test_ds_copy[idx] # {query:"",title:""}
    text_pair["pred_prob"] = prob[0] # {query:"",title:"",pred_prob:""}
    rank_result.append(text_pair) 
    print(text_pair)
for i in range(4):
    print("------------------------------------------------------------------------")
#%% （4） 排序
rank_result.sort(key=lambda x: x['pred_prob'], reverse=True) # 把按照'pred_prob'键对应的值从大到小排序
for maptext in rank_result:
    print(maptext)