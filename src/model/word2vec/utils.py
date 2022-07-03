import pandas as pd
import numpy as np


def get_train_data(train_data_path):
    return pd.read_parquet(train_data_path)


def get_embedding_w2v(w2v_model, doc_tokens):
    embeddings = []
    if len(doc_tokens) < 1:
        return np.zeros(300)
    else:
        for tok in doc_tokens:
            if tok in w2v_model.wv.index_to_key:
                embeddings.append(w2v_model.wv.word_vec(tok))
            else:
                embeddings.append(np.random.rand(300))
        return np.mean(embeddings, axis=0)
