# %%
# # 基于in-batch-negative方法的训练一个以预训练模型ERNIE1.0骨架的神经网络
import sys
sys.path.append('../')

import os
import time
import paddle
import paddlenlp
from paddlenlp.data import Tuple, Pad
from public_module.data2 import read_csv_generator, convert_example
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from functools import partial
from public_module.base_model import SemanticIndexBatchNeg

# 1、加载数据
paddle.set_device("cpu")
trainSet_path = '../literature_dataset/recall/train.csv'             #  数据集存放路径
train_ds = load_dataset(read_csv_generator, data_path=trainSet_path, lazy=False)  # 加载数据


#  2、构造训练数据的加载器
tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')

trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=64)
batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=64, shuffle=True) # 创建分布式批次采样器对象实例
batchify_fn = lambda samples, fn = Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int64'),
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int64'),
    ): [data for data in fn(samples)]

train_data_loader = paddle.io.DataLoader(dataset=train_ds.map(trans_func), batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)

# 3、加载预训练模型
pretrained_model = paddlenlp.transformers.ErnieModel.from_pretrained('ernie-1.0')
model = SemanticIndexBatchNeg(pretrained_model, margin=0.1, scale=20, output_emb_size=256)

# %%
# 4、定义模型参数，开展模型训练
epochs=3
num_training_steps = len(train_data_loader) * epochs
lr_scheduler = LinearDecayWithWarmup(5E-5, num_training_steps, 0.0)
decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=0.0,
    apply_decay_param_fun=lambda x: x in decay_params)

# %%创建保存地址
save_dir='./recall/model_param'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

global_step = 0
tic_train = time.time()

for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids = batch

        loss = model(query_input_ids=query_input_ids, title_input_ids=title_input_ids, query_token_type_ids=query_token_type_ids, title_token_type_ids=title_token_type_ids)
        global_step += 1
        if global_step % 1 == 0:
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, 10 / (time.time() - tic_train)))
            tic_train = time.time()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()

        if global_step % 10 == 0:
            save_path = os.path.join(save_dir, "model_%d" % global_step)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_param_path = os.path.join(save_path, 'model_state.pdparams')
            paddle.save(model.state_dict(), save_param_path)
            tokenizer.save_pretrained(save_path)
print('successful!')