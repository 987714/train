# %%
from functools import partial
import os
import paddle
from paddlenlp.data import Tuple, Pad
from paddlenlp.datasets import MapDataset
import paddlenlp

from public_module.base_model import SemanticIndexBase
from public_module.data2 import convert_example, gen_id2corpus
from public_module.ann_util import build_index

# %%
paddle.set_device("cpu")

# %%  1、构建模型并载入模型参数
pretrained_model = paddlenlp.transformers.AutoModel.from_pretrained("ernie-1.0")
model = SemanticIndexBase(pretrained_model, output_emb_size=256)

# 加载 模型
params_path = "model_param/model_180/model_state.pdparams"
if params_path and os.path.isfile(params_path):
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)
else:
    raise ValueError("Please set params_path with correct pretrained model file")

# 2、数据加载预处理
# %%  2.1 读取语料库
corpus_path = "../literature_dataset/recall/corpus.csv"
id2corpus = gen_id2corpus(corpus_path)

corpus_list = []
for idx, text in id2corpus.items():
    corpus_list.append({idx: text})
corpus_ds = MapDataset(corpus_list)

# %%  2.2 文本编码
tokenizer = paddlenlp.transformers.AutoTokenizer.from_pretrained("ernie-1.0")
trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=60)
# %%  2.3 批次整合
batch_sampler = paddle.io.BatchSampler(corpus_ds, batch_size=64, shuffle=False)
# %%  2.4 统一长度
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),
): [data for data in fn(samples)]
# %%  2.5 数据加载
corpus_data_loader = paddle.io.DataLoader(dataset=corpus_ds.map(trans_func), batch_sampler=batch_sampler,
                                          collate_fn=batchify_fn, return_list=True)

# 3、构建索引
output_emb_size = 256
hnsw_max_elements = 1000000
hnsw_ef = 100
hnsw_m = 100

final_index = build_index(output_emb_size, hnsw_max_elements, hnsw_ef, hnsw_m, corpus_data_loader, model)

# 创建保存文件夹
save_index_dir = "index_file"
if not os.path.exists(save_index_dir):
    os.makedirs(save_index_dir)
# 保存 ann 索引模型
save_index_path = os.path.join(save_index_dir, "final_index.bin")
final_index.save_index(save_index_path)
print('successful!')