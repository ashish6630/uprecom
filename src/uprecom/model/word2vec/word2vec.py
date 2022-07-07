from gensim.models import Word2Vec
import numpy as np
import pyarrow as pa
from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from Settings import word2vec_model_path, word2vec_embeddings_path

from uprecom.components.preprocess.lowercasefilter import LowerCaseFilter
from uprecom.components.preprocess.specialcharfilter import SpecialCharFilter
from uprecom.components.preprocess.stopword_filter import StopwordFilter
from uprecom.components.preprocess.tokenizer import Tokenizer
from uprecom.components.preprocess.whitespacefilter import WhitespaceFilter
from uprecom.model.word2vec.utils import get_train_data, get_embedding_w2v
from uprecom.myclasses.pipeline import Pipeline


class Word2VecModel:
    def __init__(
        self, min_count: int = 2, window: int = 5, sg: int = 1, workers: int = 4
    ):

        self.preprocessor_pipeline = Pipeline(
            [
                LowerCaseFilter(),
                StopwordFilter(),
                SpecialCharFilter(),
                WhitespaceFilter(),
                Tokenizer(),
            ]
        )
        self.min_count = (min_count,)
        self.window = (window,)
        self.sg = (sg,)
        self.workers = workers

    def train(self, train_data_path, model_dir_path, vectors_dir_path):
        """Train script performs the following tasks:

        - Preprocess given text data.
        - Train a word2vec model on the given model.
        - Save a parquet file with 2 columns, the given text data + its relevant document embedding.

        Document vectors in the final parquet file will be used to calculate cosine similarity and fetch
        relevant input documents for a query word during inference.

        Args:
            train_data_path: Path to a parquet file with mandatory column job description.
            model_dir_path: Path to use for saving trained word2vec model file.
            vectors_dir_path: Path to use for saving final job description + document vector parquet file.
        """

        train_df = get_train_data(train_data_path)

        if "job_ad" not in train_df.columns:
            raise ValueError("Train dataframe does not contain mandatory column job_ad")

        train_df["preprocessed_data"] = train_df["job_ad"].apply(
            self.preprocessor_pipeline.process
        )

        w2v_model = Word2Vec(
            list(train_df["preprocessed_data"]),
            min_count=2,
            window=5,
            sg=1,
            workers=4,
            vector_size=300,
        )

        train_df["vector"] = train_df["preprocessed_data"].apply(
            lambda x: get_embedding_w2v(w2v_model, x)
        )

        schema = pa.schema(
            [
                pa.field("job_ad", pa.string()),
                pa.field("vector", pa.list_(pa.float64())),
            ]
        )

        w2v_model.save(model_dir_path.as_posix())
        train_df.to_parquet(vectors_dir_path, schema=schema)

    @staticmethod
    def inference(
        query_text: str,
        model_dir_path: Path = word2vec_model_path,
        vectors_dir_path: Path = word2vec_embeddings_path,
    ) -> pd.DataFrame:
        """Returns 10 most relevant documents for a given query text.

        Args:
            query_text: Search keyword.
            model_dir_path: Location of trained word2vec model obtained after running train.
            vectors_dir_path: Location of parquet file with documents and their relevant document vectors
            obtained after running train.
        Returns:
            A dataframe with 10 entries corresponding to the 10 most relevant documents for a
            specified query text.
        """

        all_vectors = pd.read_parquet(vectors_dir_path)
        w2v_model = Word2Vec.load(model_dir_path.as_posix())

        query_text = query_text.lower()
        vector = get_embedding_w2v(w2v_model, query_text.split())
        documents = all_vectors[["job_ad"]].copy()
        documents["similarity"] = all_vectors["vector"].apply(
            lambda x: cosine_similarity(
                np.array(vector).reshape(1, -1), np.array(x).reshape(1, -1)
            ).item()
        )
        documents.sort_values(by="similarity", ascending=False, inplace=True)

        return documents.head(10).reset_index(drop=True)
