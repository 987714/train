import numpy as np
import hnswlib
from paddlenlp.utils.log import logger


def build_index(output_emb_size, hnsw_max_elements, hnsw_ef, hnsw_m, data_loader, model):

    index = hnswlib.Index(space="ip", dim=output_emb_size if output_emb_size > 0 else 768)

    index.init_index(max_elements=hnsw_max_elements, ef_construction=hnsw_ef, M=hnsw_m)
    index.set_ef(hnsw_ef)

    # Set number of threads used during batch search/construction
    # By default using all available cores
    index.set_num_threads(16)

    logger.info("start build index..........")

    all_embeddings = []

    for text_embeddings in model.get_semantic_embedding(data_loader):
        all_embeddings.append(text_embeddings.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    index.add_items(all_embeddings)

    logger.info("Total index number:{}".format(index.get_current_count()))

    return index
