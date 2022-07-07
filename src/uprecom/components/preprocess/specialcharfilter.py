import re


class SpecialCharFilter:
    def __init__(self):
        self.regex_pattern = "[^a-z]"

    def process(self, text):
        return re.sub(self.regex_pattern, " ", text)
