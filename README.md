# UPRECOM
Toy project to develop an information retrieval model. The model returns top most similar documents
for a given query text.

### Available Models

#### Word2Vec
Finds most similar documents via cosine similarity between the query text and the document embedding.

### How To Install
Uprecom uses Poetry for packaging and dependency management. If you want to build it from source,
you have to install Poetry first. This is how it can be done:

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```

```
poetry install
```

### Try Out Using Streamlit

```
strealit run src/visualise/streamlit_demo.py
```
