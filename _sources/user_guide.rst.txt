.. title:: User guide: contents

.. _user_guide:

==========
User Guide
==========

Scraping
========

The scraping module provides some simple estimator that extract meaningful
documentation from the documentation website.

:class:`~ragger_duck.scraping.APIDocExtractor` is a scraper that loads the
HTML pages and extract the documentation from the main API section. One can
provide a `chunk_size` and `chunk_overlap` to further split the documentation
sections into smaller chunks.

:class:`~ragger_duck.scraping.APINumPyDocExtractor` is a more advanced scraper
that uses `numpydoc` and it scraper to extract the documentation. Indeed, the
`numpydoc` scraper will parse the different sections and we build meaningful
chunks of documentation from the parsed sections. While, we don't control for
the chunk size, the chunks are build such that they contain information only
of a specific parameter and always refer to the class or function. We hope that
scraping in such way can remove ambiguity that could exist when building chunks
without any control. Since we rely on `numpydoc` for the parsing that expect
a specific formatting, then
:class:`~ragger_duck.scraping.APINumPyDocExtractor` is much faster than
:class:`~ragger_duck.scraping.APIDocExtractor`.

Retriever
=========

We differentiate two types of context retrievers: lexical and semantical.

Lexical retrievers
------------------

In lexical retrievers, the idea is to have exact match between the query and
the documentation.

We implement :class:`~ragger_duck.retrieval.BM25Retriever` that uses a
:class:`~sklearn.feature_extraction.text.CountVectorizer` to build the
vocabulary. Then, we use a TF-IDF weighting scheme to weight the vocabulary.
During the a query, we provide a similarity score between the query and the
each documentation chunk seen during training.

Semantical retrievers
---------------------

In semantical retrievers, the idea is to have a more flexible match between the
query and the documentation. We use an embedding model to project a document
into a vector space. During the training, these vectors are used to build a
vector database. During the query, we project the query into the vector space
and we retrieve the closest documents.

:class:`~ragger_duck.retrieval.SemanticRetriever` are using a given embedding
and an approximate nearest neighbor algorithm, namely
`FAISS <https://github.com/facebookresearch/faiss>`_.

As embedding, we provide a :class:`~ragger_duck.embedding.SentenceTransformer`
that download any pre-trained sentence transformers from HuggingFace.

Reranker: merging lexical and semantical retrievers
---------------------------------------------------

If we use both lexical and semantical retrievers, we need to merge the results
of both retrievers. :class:`~ragger_duck.retrieval.RetrieverReranker` makes
such reranking by using a cross-encoder model. In our case, cross-encoder model
is trained on Microsoft Bing query-document pairs and is available on
HuggingFace.

Prompting
=========

Prompting for API documentation
-------------------------------

:class:`~ragger_duck.prompt.APIPromptingStrategy` implements a prompting
strategy to answer API documentation questions. We get context by reranking the
search from a lexical and semantical retrievers. Once the context is retrieved,
we request a Large Language Model (LLM) to answer the question.
