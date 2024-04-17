.. _large_language_model:

=========
Prompting
=========

Prompting for API documentation
===============================

:class:`~ragger_duck.prompt.BasicPromptingStrategy` implements a prompting
strategy to answer documentation questions. We get context by reranking the
search from a lexical and semantical retrievers. Once the context is retrieved,
we request a Large Language Model (LLM) to answer the question.
