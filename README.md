# 基于 ERNIE 的文献检索与重排系统

这是一个两阶段文献搜索系统。输入自然语言查询后，系统先用向量索引从语料库中找出候选文献，再用排序模型逐条比较查询与候选，返回按相关性分数排列的结果。提供命令行和 PyQt5 桌面界面。

**技术栈：** Python · PaddlePaddle · PaddleNLP · ERNIE · hnswlib · PyQt5

## 技术实现

- **召回：** 基于 ERNIE-1.0 微调双塔式语义编码模型，使用同批次内的其他文献作为负样本。将文献编码为 256 维归一化向量，离线构建 HNSW 索引；查询时只需编码查询文本，再从索引找候选。命令行默认召回 50 条。
- **重排：** 基于 ERNIE-3.0-medium-zh 训练排序模型，用成对排序损失让相关文献的分数高于不相关文献。检索时将查询和候选文本一起输入模型，按新分数降序输出。
- **训练与评估：** 提供召回和排序的训练脚本、索引构建脚本，以及召回和排序的离线评估代码。日常搜索只加载已有产物，不重新训练。

## 本地验证

在 Windows、Python 3.12 环境中，用已有语料、权重和索引实测查询「气候变化对生物多样性的影响」。终端入口完成召回与重排；桌面版通过离屏测试验证了窗口初始化和搜索处理流程，返回 **3 条结果**，单次耗时约 **32.5 秒**（包含加载模型和索引）。这是运行验证，不是检索质量评测或平均延迟。

## 目录

| 路径 | 职责 |
| --- | --- |
| [recall/](recall/) | 召回模型微调、HNSW 建索引、批量召回与评估 |
| [rank/](rank/) | 成对排序模型训练、验证与预测 |
| [search_system/](search_system/) | 命令行搜索、PyQt5 界面和搜索处理 |
| [public_module/](public_module/) | 模型结构、数据预处理和索引构建的公共代码 |
| [showCodeSet/](showCodeSet/) | 数据读取和分词示例 |

## 本地运行

主要依赖版本记录在 [requirements.txt](requirements.txt)。请在独立 Python 环境中安装依赖；上面的验证使用的是现有 Windows 环境，不代表已验证全新安装。

```powershell
python -m pip install -r requirements.txt
```

**本仓库仅发布源码和说明。** 原始语料、训练数据、模型权重、向量索引和离线结果未随仓库发布，克隆后不能直接检索。需要自行准备有权使用的数据及与其配套的产物：

| 文件或目录 | 用途 |
| --- | --- |
| `literature_dataset/recall/corpus.csv` | 检索语料，每行一条；索引返回的编号按行序映射回文本 |
| `search_system/final_index.bin`、`search_system/recall_model_state.pdparams` | 命令行入口使用的向量索引和召回权重 |
| `recall/index_file/final_index.bin`、`recall/model_param/model_180/model_state.pdparams` | 桌面入口使用的索引和召回权重 |
| `search_system/rank_model_state.pdparams` | 两个入口共用的重排权重 |

更换语料或召回权重后，应使用同一份语料、同一行序及配套权重重建索引。首次加载 ERNIE 基础模型还需要可用的模型缓存，或能从模型源下载。

从仓库根目录运行命令行入口：

```powershell
python .\search_system\search.py
```

桌面入口依赖相对路径，需从 `search_system` 目录启动：

```powershell
cd .\search_system
python .\gui.py
```

桌面版目前在界面线程同步加载模型并执行搜索，查询过程中窗口可能短暂无响应。
