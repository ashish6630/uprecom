from src.uprecom.components.preprocess.whitespacefilter import WhitespaceFilter


def test_whitespace_scenarios():
    sentence = "this. \n\n has   too much. \n  \n whitespace"
    expected_sentence = "this. has too much. whitespace"

    filters = WhitespaceFilter()

    filtered_text = filters.process(sentence)

    assert filtered_text == expected_sentence
