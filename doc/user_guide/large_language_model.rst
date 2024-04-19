.. _large_language_model:

====================
Large Language Model
====================

In the RAG framework, the Large Language Model (LLM) is the cherry on top. It is in
charge of generating the answer to the query based on the context retrieved.

A rather important part of the LLM is related to the prompt to trigger the generation.
In this POC, we did not intend to optimize the prompt because we did not have the data
at hand to make a proper evaluation.

:class:`~ragger_duck.prompt.BasicPromptingStrategy` allows to interface the LLM with
the context found by the retriever. For prototyping purposes, we also allow the
retrievers to be bypassed. The prompt provided to the LLM is the following::

    prompt = (
        "[INST] You are a scikit-learn expert that should be able to answer"
        " machine-learning question.\n\nAnswer to the query below using the"
        " additional provided content. The additional content is composed of"
        " the HTML link to the source and the extracted contextual"
        " information.\n\nBe succinct.\n\n"
        "Make sure to use backticks whenever you refer to class, function, "
        "method, or name that contains underscores.\n\n"
        f"query: {query}\n\n{context_query} [/INST]."
    )

When bypassing the retrievers, we do not provide any context and the sentence related
to this part.
