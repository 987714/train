
from functools import partial
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.dirname(current_path)
sys.path.insert(0, current_dir)

import numpy as np
import hnswlib
import paddle
from paddlenlp.data import Tuple, Pad
from paddlenlp.datasets import MapDataset
import paddlenlp

from public_module.base_model import SemanticIndexBase, PairwiseMatching
from public_module.data2 import convert_example, create_dataloader
from public_module.data2 import gen_id2corpus, convert_pairwise_example
import copy

#预测函数
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

#%% 1、提前准备
# %%  (1) 读取语料库
# corpus_path = "../literature_dataset/recall/corpus.csv"
corpus_path = os.path.join(current_dir, r'literature_dataset/recall/corpus.csv')
id2corpus = gen_id2corpus(corpus_path)
# %%  (2) hnsw 检索网络加载
save_index_path = os.path.join(current_path, "final_index.bin" )

output_emb_size = 256
final_index = hnswlib.Index(space="ip", dim=output_emb_size) 
final_index.load_index(save_index_path)
# %% (3) 构建模型并载入模型参数
pretrained_model = paddlenlp.transformers.AutoModel.from_pretrained("ernie-1.0")
model = SemanticIndexBase(pretrained_model, output_emb_size=256)
params_path = os.path.join(current_path, "recall_model_state.pdparams")
if params_path and os.path.isfile(params_path):
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)
else:
    raise ValueError("Please set params_path with correct pretrained model file")

rank_pretrained_model = paddlenlp.transformers.AutoModel.from_pretrained("ernie-3.0-medium-zh")
rank_model = PairwiseMatching(rank_pretrained_model)
rank_params_path = os.path.join(current_path, "rank_model_state.pdparams")
if rank_params_path and os.path.isfile(rank_params_path):
    state_dict = paddle.load(rank_params_path)
    rank_model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)
else:
    raise ValueError("Please set --params_path with correct pretrained model file")

# %% （4） 数据加载器 的准备
tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained("ernie-1.0")
batchify_fn = lambda samples, fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),
    ): [data for data in fn(samples)]
trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=60)
      # 排序数据加载器 的准备
rank_tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")
rank_trans_func = partial(convert_pairwise_example, tokenizer=rank_tokenizer, max_seq_length=64, phase="predict")

#%%  2、输入文献
input_query = input("请输入需要查询的文献")
querys = []
querys.append({"text": input_query})
query_ds = MapDataset(querys)
#%%  3、召回操作-(1) 创建数据加载器
batch_sampler = paddle.io.BatchSampler(query_ds, batch_size=1, shuffle=False)
query_data_loader = paddle.io.DataLoader(dataset=query_ds.map(trans_func), batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)
#%%  3、召回操作-(2) 生成查询的嵌入向量
query_embedding = model.get_semantic_embedding(query_data_loader)

#%% 3、召回操作-(3) 用 HNSW 召回候选文档
recall_data = []
for batch_index, batch_query_embedding in enumerate(query_embedding): 
    recalled_idx, cosine_sims = final_index.knn_query(batch_query_embedding.numpy(), 50) 
    for idx, doc_idx in enumerate(recalled_idx[0]): 
        recall_data.append({"query": input_query, "title": id2corpus[doc_idx], "sim": 1.0 - cosine_sims[0][idx]})
        print( "{}\t{}\n".format(id2corpus[doc_idx], 1.0 - cosine_sims[0][idx] ) )

#%% 4、排序操作-(1) 处理候选文档
rank_ds = MapDataset(recall_data)
rank_ds_copy = copy.deepcopy(rank_ds)

rank_batch_sampler = paddle.io.BatchSampler(rank_ds_copy, batch_size=16, shuffle=False)
rank_data_loader = paddle.io.DataLoader(dataset=rank_ds_copy.map(rank_trans_func), batch_sampler=rank_batch_sampler, collate_fn=batchify_fn, return_list=True)

y_probs = predict(rank_model, rank_data_loader)
#%% 4、排序操作-(2) 预测相关性概率
rank_result = []
for idx, prob in enumerate(y_probs):
    text_pair = rank_ds[idx]
    text_pair["pred_prob"] = prob[0]
    rank_result.append(text_pair)
#%% 4、排序操作-(3) 按概率排序，返回最终结果
rank_result.sort(key=lambda x: x['pred_prob'], reverse=True)
for i in rank_result:
    print("{}\t{}".format(i['title'], i['pred_prob']))
