from gensim.parsing.preprocessing import remove_stopwords


class StopwordFilter:
    @staticmethod
    def process(text):
        return remove_stopwords(text)
