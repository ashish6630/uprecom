from uprecom.model.word2vec.word2vec import Word2VecModel
from uprecom.myclasses.enums import all_models, all_product_types


def product_recommender(
    query_text: str,
    product_type: str = "job_ad",
    model_choice: str = "word2vec",
):

    if model_choice not in all_models._member_names_:
        raise ValueError(f"{model_choice} is not a valid model name")

    if product_type not in all_product_types._member_names_:
        raise ValueError(f"{product_type} is not a supported product")

    model = Word2VecModel()

    query_result = model.inference(query_text)

    return query_result
