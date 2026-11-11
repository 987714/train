import numpy as np

# %%
def recall(rs, N=10): # [ [0,1,0000]……]
    recall_flags = [np.sum(r[0:N]) for r in rs]  # 对每个query，只召回前N个
    return np.mean(recall_flags)

# %%  1、准备「标准答案」
similar_text_pair = "../literature_dataset/recall/dev.csv"
text2similar = {}    # 创建字典
with open(similar_text_pair, "r", encoding="utf-8") as f: 
    for line in f:
        text, similar_text = line.rstrip().split("\t") # 一行：text  similar_text
        text2similar[text] = similar_text   #  {text: similar_text} 添加进字典

# %% 2、读取召回结果，生成「对错标签」
rs = []  # 列表
recall_result_file = "recall_result_file/recall_result.txt" # 召回结果存放地址
recall_num = 50  # 每个查询召回 50 个结果
with open(recall_result_file, "r", encoding="utf-8") as f: # 打开已召回结果
    relevance_labels = []   # 标签列表  （比如[1,0,0,...]）
    for index, line in enumerate(f): 
        if index % recall_num == 0 and index != 0:        # 每个查询语句，对应recall_num条召回
            rs.append(relevance_labels)                   # relevance_labels数量满足recall_num条，存入rs
            relevance_labels = []                         # 清空relevance_labels

        query, recalled_text, cosine_sim = line.rstrip().split("\t")
        if text2similar[query] == recalled_text:         #  答案匹配一致
            relevance_labels.append(1)   # 标签数组：存入1
        else:
            relevance_labels.append(0)   # 标签数组：存入0

# %%  3、 计算召回率  Recall@N
recalls = [1, 5, 10, 20, 50]  # N的取值选择
recall_N = []                 # 存放计算得到的召回率
for topN in recalls:          # 枚举N的每个值
    R = round(100 * recall(rs, N=topN), 3) # 调用recall计算召回率，乘 100 转成百分比，保留 3 位小数
    recall_N.append(R)        # 添加入recall_N

# %% 4、 保存结果
result_tsv_file = "recall_result_file/result.tsv"   # 存放输出结果地址
with open(result_tsv_file, "w", encoding="utf-8") as f:  # 打开，只写
    res = []
    for i in range(len(recalls)): 
        N = recalls[i] 
        recall_val = recall_N[i] 
        print("recall@{}={}".format(N, recall_val)) # 打印结果
        res.append(str(recall_val))    #转为字符串类型，存入res列表
    
    f.write("\t".join(res) + "\n")    # 写入文件；join用法为：字符串=‘分隔符’.join(列表)。