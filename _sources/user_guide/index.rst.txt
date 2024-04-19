.. title:: User guide

.. _user_guide:

==========
User Guide
==========

Before to go in depth into implementation details of the different components of our
Retrieval Augmented Generation (RAG) framework, we provide a high-level overview of
the main components.

.. _intro_rag:

What is Retrieval Augmented Generation?
=======================================

Before to go in more details regarding Retrieval Augmented Generation (RAG), let's
first define the framework in which we will use a large language model (LLM). The
graphic below represents the interaction between our user and the LLM.

.. image:: /_static/img/diagram/llm.png
    :width: 100%
    :align: center
    :class: transparent-image

In this proof-of-concept (POC), we are interested in a "zero-shot" setting. It means
that we expect our user to formulate a question in natural language and the LLM will
generate an answer.

The way to query the LLM can be done in two ways: (i) through an API such as when using
GPT-* from OpenAI or (ii) by locally running the model using open-weight models such
as Mistral or LLama.

Now, let's introduce the RAG framework.

.. image:: /_static/img/diagram/rag.png
    :width: 100%
    :align: center
    :class: transparent-image

The major difference with the previous framework is an additional step that consists
in retrieving relevant information from a given source of information before answering
the user's query. The retrieved information is provided as a context to the LLM during
prompting, and the LLM will therefore generate an answer conditioned on this context.

It should be noted that information retrieval is not a new concept and has been
extensively studied in the past and it is also related to the application of search
engine. In the next section, we will go in more details into the information retrieval
components when used for a RAG framework.

.. _intro_info_retrieval:

Information retrieval
=====================

Concepts
--------

Before to explain how a retriever is trained, we first show the main components of
such retriever.

.. image:: /_static/img/diagram/retrieval_phase.png
    :width: 100%
    :align: center
    :class: transparent-image

A retriever has two main components: (i) an algorithm to transform natural text into
a mathematical vector representation and (ii) a database containing vectors and their
corresponding natural text. This database is also capable of finding the most similar
vectors to a given query vector.

During the training phase, a source of information containing natural text is used to
build a set of vector representations. These vectors are used to populate the database.

During the retrieval phase, a user's query is passed to the algorithm to create a vector
representation. Then, the most similar vectors are found in the database and the
corresponding natural texts are returned. Those documents are then used as context for
the LLM in the previous RAG framework.

Details regarding the retrievers
--------------------------------

In this section, we provide a couple of details regarding the retrievers. However,
our reader can refer to the following comprehensive review for more details [1]_.

Without going into the details, we can distinguish two types of retrievers: (i)
lexical retrievers based on the Bag-of-Words (BoW) model and (ii) semantic retrievers
based on neural networks.

Lexical retrievers are based on word counts in documents and queries. They are simple
but they lack the ability to capture the meaning of words. Several approaches have been
proposed to improve the performance of these retrievers by expanding queries, documents
or modeling topics. Those retrievers create a sparse representation that can be
leveraged to find the most similar documents through inverted indexes.

Semantic retrievers are based on neural networks and project the text into a continuous
vector space. While they are better at capturing the meaning of words, they make the
search more complex due to the dense representation. In this case, approximate nearest
neighbors algorithms are used to find the most similar documents.

References
----------

.. [1] Guo, J., Cai, Y., Fan, Y., Sun, F., Zhang, R., & Cheng, X. (2022).
       "Semantic models for the first-stage retrieval: A comprehensive review."
       ACM Transactions on Information Systems (TOIS), 40(4), 1-42.
       https://arxiv.org/abs/2103.04831

Implementation details
======================

In the previous sections, we presented the general ideas behind the RAG framework.
However, the devil is in the details. In the following sections, we present some
implementation details regarding some inner steps of the RAG framework that are
important if you want to get meaningful results.

.. toctree::
    :maxdepth: 2

    text_scraping
    information_retrieval
    large_language_model
