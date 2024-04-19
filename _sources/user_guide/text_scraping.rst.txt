.. _text_scraping:

=============
Text Scraping
=============

In a Retrieval Augmented Generation (RAG) framework, the "document" retrieved and
provided to the Large Language Model (LLM) to generate an answer corresponds to chunks
extracted from the documentation.

The first important aspect is to be aware that the context of the LLM is limited.
Therefore, we need to provide chunks of documentation that are relatively limited and
focused to not reach the context limit.

The most common strategy is therefore to extract chunks of text with given number of
token and overlap between chunks.

.. image:: /_static/img/diagram/naive_chunks.png
    :width: 100%
    :align: center
    :class: transparent-image

The various tutorial to build RAG models are using this strategy. While it is a fast
way to get started, it is not the best strategy to get the most of the scikit-learn
documentation. In the subsequent sections, we present different strategies specifically
designed for some portion of the scikit-learn documentation.

API documentation
=================

We refer to "API documentation" to the following documentation entry point:
https://scikit-learn.org/stable/modules/classes.html.

It corresponds to the documentation of each class and function implemented in
scikit-learn. This documentation is automatically generated from the docstrings
of the classes and functions. These docstrings follow the `numpydoc` formatting.
As an example, we show a generated HTML page containing the documentation of a
scikit-learn estimator:

.. image:: /_static/img/diagram/api_doc_generated_html.png
    :width: 100%
    :align: center
    :class: transparent-image

Before diving into the chunking mechanism, it is interesting to think about the type
of queries that such documentation can help at answering. Indeed, these documentation
pages are intended to provide information about class or function parameters, short
usage snippet of code and related classes or functions. The narration on these pages
are relatively short and further discussions are generally provided in the user guide
instead. So we would expect that the chunks of documentation to be useful to answer
questions as:

- What are the parameters of `LogisticRegression`?
- What are the values of the `strategy` parameter in a dummy classifier?

:class:`~ragger_duck.scraping.APINumPyDocExtractor` is a more advanced scraper
that uses `numpydoc` and it scraper to extract the documentation. Indeed, the
`numpydoc` scraper will parse the different sections and we build meaningful
chunks of documentation from the parsed sections. While, we don't control for
the chunk size, the chunks are build such that they contain information only
of a specific parameter and always refer to the class or function. We hope that
scraping in such way can remove ambiguity that could exist when building chunks
without any control.

User Guide documentation
========================

:class:`~ragger_duck.scraping.UserGuideDocExtractor` is a scraper that extract
documentation from the user guide. It is a simple scraper that extract
text information from the webpage. Additionally, this text can be chunked.
