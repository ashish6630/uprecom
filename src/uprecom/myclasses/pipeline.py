from typing import List


class Pipeline:
    def __init__(self, processors: List):
        self.pipeline_components = processors

    def process(self, text):
        for processor in self.pipeline_components:
            text = processor.process(text)
        return text
