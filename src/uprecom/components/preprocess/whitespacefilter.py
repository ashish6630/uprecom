import re


class WhitespaceFilter:
    def __init__(self):
        ...

    def process(self, text: str) -> str:
        """Returns text without whitespaces."""
        pattern = re.compile(r"\s+")
        return re.sub(pattern, " ", text)
