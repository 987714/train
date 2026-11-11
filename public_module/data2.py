import os
import paddle
from paddlenlp.utils.log import logger

# 编写： 预训练模型的微调过程中，训练集的读取生成器函数
def read_csv_generator(data_path):
  with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if len(data) != 2:
                continue
            yield {'question': data[0], 'answer': data[1]}

def create_dataloader(dataset, mode="train", batch_size=1, batchify_fn=None, trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)
    shuffle = True if mode == "train" else False
    if mode == "train":
        batch_sampler = paddle.io.DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=batchify_fn, return_list=True)

# 文本编码：将文本转换为模型可理解的离散符号（Token ID）
def convert_example(example, tokenizer, max_seq_length=512, pad_to_max_seq_len=False):
    result = []
    for key, text in example.items():
        encoded_inputs = tokenizer(text=text, max_seq_len=max_seq_length, pad_to_max_seq_len=pad_to_max_seq_len)
        input_ids = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        result += [input_ids, token_type_ids]
    return result

def gen_id2corpus(corpus_file):
    id2corpus = {}
    with open(corpus_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            id2corpus[idx] = line.rstrip()
    return id2corpus


def read_train(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        flag = 0
        for line in f:
            if flag != 0:
                data = line.rstrip().split("\t")
                if len(data) != 3:
                    continue
                yield {'query': data[0], 'title': data[1], 'neg_title': data[2]}

            flag = 1

# %%
def read_dev(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        flag = 0
        for line in f:
            if flag != 0:
                data = line.rstrip().split("\t")
                if len(data) != 3:
                    continue
                yield {'query': data[0], 'title': data[1], 'label': data[2]}

            flag = 1

def read_text_pair(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = line.rstrip().split("\t")
            if len(data) != 3:
                continue
            yield {"query": data[0], "title": data[1]}

def convert_pairwise_example(example, tokenizer, max_seq_length=512, phase="train"):

    if phase == "train":
        query, pos_title, neg_title = example["query"], example["title"], example["neg_title"]

        pos_inputs = tokenizer(text=query, text_pair=pos_title, max_seq_len=max_seq_length)
        neg_inputs = tokenizer(text=query, text_pair=neg_title, max_seq_len=max_seq_length)

        pos_input_ids = pos_inputs["input_ids"]
        pos_token_type_ids = pos_inputs["token_type_ids"]

        neg_input_ids = neg_inputs["input_ids"]
        neg_token_type_ids = neg_inputs["token_type_ids"]

        return [pos_input_ids, pos_token_type_ids, neg_input_ids, neg_token_type_ids]

    else:
        query, title = example["query"], example["title"]

        inputs = tokenizer(text=query, text_pair=title, max_seq_len=max_seq_length)

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]

        if phase == "eval":
            return input_ids, token_type_ids, example["label"]
        elif phase == "predict":
            return input_ids, token_type_ids
        else:
            raise ValueError("not supported phase:{}".format(phase))