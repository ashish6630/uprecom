from pathlib import Path
import click

from Settings import job_data_train_path, word2vec_model_path, word2vec_embeddings_path
from uprecom.model.word2vec.word2vec import Word2VecModel


@click.command()
@click.option(
    "--mode",
    type=str,
    default="train",
    help="Specify whether" "model is train or test.",
)
@click.option(
    "--input_data_path",
    type=Path,
    default=job_data_train_path,
    help="Train data as a parquet file.",
)
@click.option(
    "--model_dir_path",
    type=Path,
    default=word2vec_model_path,
    help="Where to save trained model.",
)
@click.option(
    "--vectors_dir_path",
    type=Path,
    default=word2vec_embeddings_path,
    help="Where to save trained embeddings",
)
def word2vec(mode: str, input_data_path, model_dir_path, vectors_dir_path):
    if mode == "train":
        model = Word2VecModel()
        model.train(
            train_data_path=input_data_path,
            model_dir_path=model_dir_path,
            vectors_dir_path=vectors_dir_path,
        )


if __name__ == "__main__":
    word2vec()
