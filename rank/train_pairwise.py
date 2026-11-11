# %%
import sys
sys.path.append('../')
import os
import time
from functools import partial

import numpy as np
import paddle
from public_module.data2 import convert_pairwise_example
from public_module.data2 import read_train, read_dev
from public_module.base_model import PairwiseMatching

from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModel, AutoTokenizer, LinearDecayWithWarmup

# %%

@paddle.no_grad()
def evaluate(model, metric, data_loader):
    model.eval()
    metric.reset()

    for idx, batch in enumerate(data_loader):
        input_ids, token_type_ids, labels = batch
        pos_probs = model.predict(input_ids=input_ids, token_type_ids=token_type_ids)
        neg_probs = 1.0 - pos_probs
        preds = np.concatenate((neg_probs, pos_probs), axis=1)

        metric.update(preds=preds, labels=labels)
        auc = metric.accumulate()

    print("phase: dev, auc: {:.3}".format(auc))
    metric.reset()
    model.train()

# 1、 构建 训练集、验证集数据加载器
# %%  (1) 读取数据
train_path = "../literature_dataset/rank/train_pairwise.csv"
dev_path = "../literature_dataset/rank/dev_pairwise.csv"
train_ds = load_dataset(read_train, data_path=train_path, lazy=False)
dev_ds = load_dataset(read_dev, data_path=dev_path, lazy=False)
# %%  (2) 文本编码
tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")
trans_func_train = partial(convert_pairwise_example, tokenizer=tokenizer, max_seq_length=128, phase="train")
trans_func_eval = partial(convert_pairwise_example, tokenizer=tokenizer, max_seq_length=128, phase="eval")
# %%  (3) 批次整合
batch_sampler_train = paddle.io.DistributedBatchSampler(train_ds, batch_size=32, shuffle=True)
batch_sampler_dev = paddle.io.BatchSampler(dev_ds, batch_size=32, shuffle=False)
# %%  (4) 统一长度
batchify_fn_train = lambda samples, fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),
    ): [data for data in fn(samples)]
batchify_fn_eval = lambda samples, fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),
        Stack(dtype="int64"),
    ): [data for data in fn(samples)]
# %%  (5) 数据加载
train_data_loader = paddle.io.DataLoader(dataset=train_ds.map(trans_func_train), batch_sampler=batch_sampler_train, collate_fn=batchify_fn_train, return_list=True)
dev_data_loader = paddle.io.DataLoader(dataset=dev_ds.map(trans_func_eval), batch_sampler=batch_sampler_dev, collate_fn=batchify_fn_eval, return_list=True)

#%% 2、加载预训练模型
pretrained_model = AutoModel.from_pretrained("ernie-3.0-medium-zh")
model = PairwiseMatching(pretrained_model, margin=0.1) 

#%% 3、定义模型参数，开展模型训练
epochs = 3
num_training_steps = len(train_data_loader) * epochs
lr_scheduler = LinearDecayWithWarmup(2E-5, num_training_steps, 0.0)
decay_params = [
    p.name for n, p in model.named_parameters() 
    if not any(nd in n for nd in ["bias", "norm"])
]

optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=0.0,
    apply_decay_param_fun=lambda x: x in decay_params)

#%% 4、 下面正式开始训练模型
save_dir="model_param"
global_step = 0
tic_train = time.time()

metric = paddle.metric.Auc()
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1): 
        pos_input_ids, pos_token_type_ids, neg_input_ids, neg_token_type_ids = batch

        loss = model(
            pos_input_ids=pos_input_ids,
            neg_input_ids=neg_input_ids,
            pos_token_type_ids=pos_token_type_ids,
            neg_token_type_ids=neg_token_type_ids,
        ) 

        global_step += 1

        if global_step % 10 == 0: 
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, 10 / (time.time() - tic_train)))
            tic_train = time.time()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()

        if global_step % 100 == 0:
            evaluate(model, metric, dev_data_loader)
            
            save_path = os.path.join(save_dir, "model_%d" % global_step) 
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_param_path = os.path.join(save_path, "model_state.pdparams") 
            paddle.save(model.state_dict(), save_param_path)
            tokenizer.save_pretrained(save_path)

#%% 5、下面是把保存好的模型参数文件加载到一个新的模型中，并进行性能测试的过程
model_2 = PairwiseMatching(pretrained_model, margin=0.1)
params_path = "model_param/model_400/model_state.pdparams"
state_dict = paddle.load(params_path)
model_2.set_dict(state_dict)
evaluate(model_2, metric, dev_data_loader)