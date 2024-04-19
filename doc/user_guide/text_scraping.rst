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

The most common strategy is to extract chunks of text with a given number of tokens and
an overlap between chunks.

.. image:: /_static/img/diagram/naive_chunks.png
    :width: 100%
    :align: center
    :class: transparent-image

The various tutorials to build RAG models use this strategy. While it is a fast way to
get started, it is not the best strategy to get the most out of the scikit-learn
documentation. In the subsequent sections, we present different strategies
specifically designed for certain portions of the scikit-learn documentation.

API documentation
=================

We refer to "API documentation" to the following documentation entry point:
https://scikit-learn.org/stable/modules/classes.html.

It corresponds to the documentation of each class and function implemented in
scikit-learn. This documentation is automatically generated from the docstrings of the
classes and functions. These docstrings follow the `numpydoc` formatting. As an example,
we show a generated HTML page containing the documentation of a scikit-learn estimator:

.. image:: /_static/img/diagram/api_doc_generated_html.png
    :width: 100%
    :align: center
    :class: transparent-image

Before diving into the chunking mechanism, it is interesting to think about the type of
queries that such documentation can help at answering. Indeed, these documentation pages
are intended to provide information about class or function parameters, short usage
snippet of code and related classes or functions. The narration on these pages are
relatively short and further discussions are generally provided in the user guide
instead. So we would expect that the chunks of documentation to be useful to answer
questions as:

- What are the parameters of `LogisticRegression`?
- What are the values of the `strategy` parameter in a dummy classifier?

So now that we better framed our expectations, we can think about the chunks extraction.
We could go forward with the naive approach described above. However, it will fall sort
to help the LLM to answer the questions. Let's go into an example to illustrate this
point.

Let's consider the second question above: "What are the values of the `strategy`
parameter in a dummy classifier?". While our retrievers (:ref:`information_retrieval`)
are able to get the association between the `DummyClassifier` and the `strategy`
parameter, the LLM will not be able to get this link if the chunk retrieved does not
contain this relationship. Indeed, the naive approach will provide a chunk where
`strategy` could be mentioned but it might not belong to the `DummyClassifier` class.

For instance, we could retrieve the following three chunks that are relatively relevant
to the query:

**Chunk #1**::

    strategy : {"most_frequent", "prior", "stratified", "uniform", \
                "constant"}, default="prior"
            Strategy to use to generate predictions.

            * "most_frequent": the `predict` method always returns the most
              frequent class label in the observed `y` argument passed to `fit`.
              The `predict_proba` method returns the matching one-hot encoded
              vector.
            * "prior": the `predict` method always returns the most frequent
              class label in the observed `y` argument passed to `fit` (like
              "most_frequent"). ``predict_proba`` always returns the empirical
              class distribution of `y` also known as the empirical class prior
              distribution.
            * "stratified": the `predict_proba` method randomly samples one-hot
              vectors from a multinomial distribution parametrized by the empirical
              class prior probabilities.
              The `predict` method returns the class label which got probability
              one in the one-hot vector of `predict_proba`.
              Each sampled row of both methods is therefore independent and
              identically distributed.
            * "uniform": generates predictions uniformly at random from the list
              of unique classes observed in `y`, i.e. each class has equal
              probability.
            * "constant": always predicts a constant label that is provided by
              the user. This is useful for metrics that evaluate a non-majority
              class.

**Chunk #2**::

    strategy : {"mean", "median", "quantile", "constant"}, default="mean"
            Strategy to use to generate predictions.

            * "mean": always predicts the mean of the training set
            * "median": always predicts the median of the training set
            * "quantile": always predicts a specified quantile of the training set,
              provided with the quantile parameter.
            * "constant": always predicts a constant value that is provided by
              the user.

**Chunk #3**::

    strategy : str, default='mean'
        The imputation strategy.

        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
          If there is more than one such value, only the smallest is returned.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.

So the chunks are relevant to the `strategy` parameter but they are related to
`DummyClassifier`, `DummyRegressor` and `SimpleImputer` classes.

If we provide such information to a human that is not aware about the scikit-learn API,
they will not be able to know which of the above chunks is relevant to answer to the
query. If they are experts, then they might use their previous knowledge to select the
relevant chunk.

So when it comes an LLM, you should not be expecting more than a human: if the LLM has
been trained on some similar queries, then it might be able to use the relevant
information but otherwise, it will not be the case. As an example, Mistral 7b model
would only summarize the information of the chunks and provide a really unhelpful
answer.

As a straightforward solution to the above problem, we could think that we should go
beyond the naive chunking strategy. For instance, if our chunk would contain the
associated class or function to the parameter description, then it will allow to
disambiguate the information and thus help our LLM to answer the
relevant question.

As previously stated, scikit-learn use `numpydoc` formalism to document the classes and
functions. This library comes with a parser that structures the docstring information
such that you know about the section, the parameters, the types, etc. We implemented
:class:`~ragger_duck.scraping.APINumPyDocExtractor` that leverages this information to
build meaningful chunks of documentation. The chunk size in this case is not controlled
but because of the nature of the documentation, we know that it will never be too large.

As an example, a chunk created and that is going to be relevant to the previous query is
the following::

    source: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    content: Parameter strategy of sklearn.dummy.DummyClassifier.
    strategy is described as 'Strategy to use to generate predictions.

    * "most_frequent": the `predict` method always returns the most
      frequent class label in the observed `y` argument passed to `fit`.
      The `predict_proba` method returns the matching one-hot encoded
      vector.
    * "prior": the `predict` method always returns the most frequent
      class label in the observed `y` argument passed to `fit` (like
      "most_frequent"). ``predict_proba`` always returns the empirical
      class distribution of `y` also known as the empirical class prior
      distribution.
    * "stratified": the `predict_proba` method randomly samples one-hot
      vectors from a multinomial distribution parametrized by the empirical
      class prior probabilities.
      The `predict` method returns the class label which got probability
      one in the one-hot vector of `predict_proba`.
      Each sampled row of both methods is therefore independent and
      identically distributed.
    * "uniform": generates predictions uniformly at random from the list
      of unique classes observed in `y`, i.e. each class has equal
      probability.
    * "constant": always predicts a constant label that is provided by
      the user. This is useful for metrics that evaluate a non-majority
      class.

    .. versionchanged:: 0.24
        The default value of `strategy` has changed to "prior" in version
        0.24.' and has the following type(s): {"most_frequent", "prior", "stratified",
        "uniform", "constant"}, default="prior"

In some ways, we tried to build chunks keeping a sort of relationship. In this case, by
providing still the three previous chunks but keeping the class relationship, a Mistral
7b model is able to disambiguate the information and provide a relevant answer.

User Guide documentation
========================

:class:`~ragger_duck.scraping.UserGuideDocExtractor` is a scraper that extract
documentation from the user guide. It is a simple scraper that extract
text information from the webpage. Additionally, this text can be chunked.
