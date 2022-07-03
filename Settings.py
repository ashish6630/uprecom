from pathlib import Path

HOME = Path(__file__).parent

job_data_train_path = HOME / 'data/word2vec/train/job_data.parquet'
word2vec_model_path = HOME / 'data/word2vec/model/w2v.model'
word2vec_embeddings_path = HOME / 'data/word2vec/model/final_vectors.parquet'

