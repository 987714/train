# %%
from functools import partial
import os
import paddle
from paddlenlp.data import Tuple, Pad
from paddlenlp.datasets import MapDataset
import paddlenlp

import hnswlib
from public_module.base_model import SemanticIndexBase
from public_module.data2 import convert_example
from public_module.data2 import gen_id2corpus
# %%
paddle.set_device("cpu")

# 下面获取验证数据集中的所有query
def get_query_text(similar_text_pair_file):
    querys = []
    with open(similar_text_pair_file, "r", encoding="utf-8") as f:
        for line in f:
            splited_line = line.rstrip().split("\t")
            if len(splited_line) != 2:
                continue
            if not splited_line[0] or not splited_line[1]:
                continue
            querys.append({"text": splited_line[0]})
    return querys

# %%  (1) 读取语料库
corpus_path = "../literature_dataset/recall/corpus.csv"
id2corpus = gen_id2corpus(corpus_path)

# %%  (2) hnsw 检索网络加载
save_index_path = "index_file/final_index.bin"
output_emb_size = 256
final_index = hnswlib.Index(space="ip", dim=output_emb_size)
final_index.load_index(save_index_path)

# %% (3) 构建模型并载入模型参数
pretrained_model = paddlenlp.transformers.AutoModel.from_pretrained("ernie-1.0")
model = SemanticIndexBase(pretrained_model, output_emb_size=256)

params_path = "model_param/model_180/model_state.pdparams"
if params_path and os.path.isfile(params_path): 
    state_dict = paddle.load(params_path) 
    model.set_dict(state_dict) 
    print("Loaded parameters from %s" % params_path) 
else:
    raise ValueError("Please set params_path with correct pretrained model file")

# %% （4） 加载数据，构建数据加载器
similar_text_pair_path = "../literature_dataset/recall/dev.csv"
query_list = get_query_text(similar_text_pair_path)
query_ds = MapDataset(query_list) 

tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained("ernie-1.0")
trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=60)

batch_sampler = paddle.io.BatchSampler(query_ds, batch_size=64, shuffle=False)
batchify_fn = lambda samples, fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),
    ): [data for data in fn(samples)]
query_data_loader = paddle.io.DataLoader(dataset=query_ds.map(trans_func), batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)

query_embedding = model.get_semantic_embedding(query_data_loader)

# %% (5) 下面正式开始召回
recall_result_dir = "recall_result_file"
if not os.path.exists(recall_result_dir): 
    os.mkdir(recall_result_dir)

recall_result_file = "recall_result.txt"
recall_result_file = os.path.join(recall_result_dir, recall_result_file) 

with open(recall_result_file, "w", encoding="utf-8") as f: 
    for batch_index, batch_query_embedding in enumerate(query_embedding):
        recalled_idx, cosine_sims = final_index.knn_query(batch_query_embedding.numpy(), 50) 

        batch_size = len(cosine_sims)

        for row_index in range(batch_size):
            text_index = 64 * batch_index + row_index 
            for idx, doc_idx in enumerate(recalled_idx[row_index]):
                f.write( "{}\t{}\t{}\n".format(query_list[text_index]["text"], id2corpus[doc_idx], 1.0 - cosine_sims[row_index][idx] ) )