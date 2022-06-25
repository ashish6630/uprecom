import pytest

from src.components.preprocess.whitespace import WhitespaceFilter
from pytest_lazyfixture import lazy_fixture

@pytest.fixture
def whitespacefilter():
    myfilter = WhitespaceFilter()
    return myfilter


@pytest.mark.parametrize(
    "text, expected_return, filter_fixture",
    [
        (
            "this text. \n\n has   too much. \n  \n whitespace",
            "this text. has too much. whitespace",
            lazy_fixture('whitespacefilter')
        ),
        (
            "Lorem ipsum dolor. \n\n sit amet, consectetur. \n\n adipisici elit",
            "Lorem ipsum dolor. sit amet, consectetur. adipisici elit",
            lazy_fixture('whitespacefilter')
        )
    ]
)
def test_whitespace_scenarios(text, expected_return, filter_fixture):
    filtered_text = filter_fixture.process(text)
    assert filtered_text == expected_return
