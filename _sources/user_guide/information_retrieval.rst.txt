.. _information_retrieval:

=========
Retriever
=========

We differentiate two types of context retrievers: lexical and semantical.

Lexical retrievers
==================

In lexical retrievers, the idea is to have exact match between the query and
the documentation.

We implement :class:`~ragger_duck.retrieval.BM25Retriever` that uses a
:class:`~sklearn.feature_extraction.text.CountVectorizer` to build the
vocabulary. Then, we use a TF-IDF weighting scheme to weight the vocabulary.
During the a query, we provide a similarity score between the query and the
each documentation chunk seen during training.

Semantical retrievers
=====================

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
===================================================

If we use both lexical and semantical retrievers, we need to merge the results
of both retrievers. :class:`~ragger_duck.retrieval.RetrieverReranker` makes
such reranking by using a cross-encoder model. In our case, cross-encoder model
is trained on Microsoft Bing query-document pairs and is available on
HuggingFace.
