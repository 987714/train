from functools import partial
import os
import sys
sys.path.append('..')
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

def Search(input_query, recall_num):
    ## 1、 召回准备阶段：
    # %% （1） 语料库加载
    corpus_path = "../literature_dataset/recall/corpus.csv"
    id2corpus = gen_id2corpus(corpus_path)

    # %% （2） hnsw 检索网络加载
    save_index_path = "../recall/index_file/final_index.bin"
    output_emb_size = 256
    final_index = hnswlib.Index(space="ip", dim=output_emb_size)
    final_index.load_index(save_index_path)

    # %% (3) 构建模型并载入模型参数
    pretrained_model = paddlenlp.transformers.AutoModel.from_pretrained("ernie-1.0")
    model = SemanticIndexBase(pretrained_model, output_emb_size=256)
    params_path = "../recall/model_param/model_180/model_state.pdparams"
    # params_path = "model_state.pdparams"
    if params_path and os.path.isfile(params_path):
        state_dict = paddle.load(params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % params_path)
    else:
        raise ValueError("Please set params_path with correct pretrained model file")

    # 2、 排序准备阶段：
    #  %% （1） 加载排序模型
    rank_pretrained_model = paddlenlp.transformers.AutoModel.from_pretrained("ernie-3.0-medium-zh")
    rank_model = PairwiseMatching(rank_pretrained_model)

    rank_params_path = "rank_model_state.pdparams"
    if rank_params_path and os.path.isfile(rank_params_path):
        state_dict = paddle.load(rank_params_path)
        rank_model.set_dict(state_dict)
        print("Loaded parameters from %s" % params_path)
    else:
        raise ValueError("Please set --params_path with correct pretrained model file")

    # %%
    rank_tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")
    rank_trans_func = partial(convert_pairwise_example, tokenizer=rank_tokenizer, max_seq_length=64, phase="predict")

    # 3、输入查询语句
    querys = []
    querys.append({"text": input_query})
    query_ds = MapDataset(querys)

    tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained("ernie-1.0")
    trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=60)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),
    ): [data for data in fn(samples)]
    batch_sampler = paddle.io.BatchSampler(query_ds, batch_size=1, shuffle=False)
    query_data_loader = paddle.io.DataLoader(dataset=query_ds.map(trans_func), batch_sampler=batch_sampler,
                                             collate_fn=batchify_fn, return_list=True)

    # 4、下面正式开始召回
    query_embedding = model.get_semantic_embedding(query_data_loader)

    recall_data = []
    for batch_index, batch_query_embedding in enumerate(query_embedding):
        recalled_idx, cosine_sims = final_index.knn_query(batch_query_embedding.numpy(), recall_num)
        for idx, doc_idx in enumerate(recalled_idx[0]):
            recall_data.append({"query": input_query, "title": id2corpus[doc_idx], "sim": 1.0 - cosine_sims[0][idx]})

    # 5、下面正式开始排序
    rank_ds = MapDataset(recall_data)
    rank_ds_copy = copy.deepcopy(rank_ds)

    rank_batch_sampler = paddle.io.BatchSampler(rank_ds_copy, batch_size=16, shuffle=False)
    rank_data_loader = paddle.io.DataLoader(dataset=rank_ds_copy.map(rank_trans_func), batch_sampler=rank_batch_sampler,
                                            collate_fn=batchify_fn, return_list=True)

    y_probs = predict(rank_model, rank_data_loader)
    rank_result = []
    for idx, prob in enumerate(y_probs):
        text_pair = rank_ds[idx]
        text_pair["pred_prob"] = prob[0]
        rank_result.append(text_pair)
    rank_result.sort(key=lambda x: x['pred_prob'], reverse=True)

    return rank_result