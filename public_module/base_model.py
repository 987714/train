import abc
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class SemanticIndexBase(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, output_emb_size=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.output_emb_size = output_emb_size

        if output_emb_size > 0:
            self.emb_reduce_linear = paddle.nn.Linear(768, output_emb_size) 

    def get_pooled_embedding(self,input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)

        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(cls_embedding)

        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1) 

        return cls_embedding

    def get_semantic_embedding(self, data_loader): 
        self.eval()
        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                input_ids = paddle.to_tensor(input_ids)
                token_type_ids = paddle.to_tensor(token_type_ids)

                text_embeddings = self.get_pooled_embedding(input_ids, token_type_ids=token_type_ids)

                yield text_embeddings


class SemanticIndexBatchNeg(SemanticIndexBase):
    def __init__(self, pretrained_model, dropout=None, margin=0.3, scale=30, output_emb_size=None):
        super().__init__(pretrained_model, dropout, output_emb_size)

        self.margin = margin
        self.scale = scale

    def forward(self, query_input_ids,    title_input_ids,    query_token_type_ids=None, query_position_ids=None, query_attention_mask=None,    title_token_type_ids=None, title_position_ids=None, title_attention_mask=None):
        query_cls_embedding = self.get_pooled_embedding(query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask)

        title_cls_embedding = self.get_pooled_embedding(title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask)

        cosine_sim = paddle.matmul(query_cls_embedding, title_cls_embedding, transpose_y=True)

        margin_diag = paddle.full(shape=[query_cls_embedding.shape[0]], fill_value=self.margin, dtype="float32")

        cosine_sim = cosine_sim - paddle.diag(margin_diag)

        cosine_sim = cosine_sim * self.scale

        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype='int64')
        labels = paddle.reshape(labels, shape=[-1, 1])

        loss = F.cross_entropy(input=cosine_sim, label=labels)

        return loss
class PairwiseMatching(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, margin=0.1):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.margin = margin

        self.similarity = nn.Linear(self.ptm.config["hidden_size"], 1)

    def predict(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)
        cls_embedding = self.dropout(cls_embedding)
        sim_score = self.similarity(cls_embedding)
        sim_score = F.sigmoid(sim_score)

        return sim_score

    def forward(
            self,
            pos_input_ids,
            neg_input_ids,

            pos_token_type_ids=None,
            neg_token_type_ids=None,

            pos_position_ids=None,
            neg_position_ids=None,

            pos_attention_mask=None,
            neg_attention_mask=None,
    ):
        _, pos_cls_embedding = self.ptm(pos_input_ids, pos_token_type_ids, pos_position_ids, pos_attention_mask)
        _, neg_cls_embedding = self.ptm(neg_input_ids, neg_token_type_ids, neg_position_ids, neg_attention_mask)

        pos_embedding = self.dropout(pos_cls_embedding)
        neg_embedding = self.dropout(neg_cls_embedding)

        pos_sim = self.similarity(pos_embedding)
        neg_sim = self.similarity(neg_embedding)

        pos_sim = F.sigmoid(pos_sim)
        neg_sim = F.sigmoid(neg_sim)

        labels = paddle.full(shape=[pos_cls_embedding.shape[0]], fill_value=1.0, dtype="float32")

        loss = F.margin_ranking_loss(pos_sim, neg_sim, labels, margin=self.margin)

        return loss



