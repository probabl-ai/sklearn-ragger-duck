import logging

from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import HasMethods

logger = logging.getLogger(__name__)


class BasicPromptingStrategy(BaseEstimator):
    """Prompting strategy for answering a query.

    Once we retrieve the context, we request to answer the query using the context.
    We allow to not use the retrieved context to answer the query.

    Parameters
    ----------
    llm : llm instance
        The language model to use for the prompting. We expect the model to implement
        a `__call__` method that takes a prompt and returns a response. It should be an
        "Instruct"-based model.

    retriever : retriever instance
        The retriever to use for retrieving the context. We expect the retriever to
        implement a `query` method.

    use_retrieved_context : bool, default=True
        Whether to use the retriever to retrieve the context before prompting.
    """

    _parameter_constraints = {
        "llm": [HasMethods(["__call__"])],
        "retriever": [HasMethods(["query"])],
        "use_retriever": [bool],
    }

    def __init__(self, *, llm, retriever, use_retrieved_context=True):
        self.llm = llm
        self.retriever = retriever
        self.use_retrieved_context = use_retrieved_context

    def __call__(self, query, **prompt_kwargs):
        logger.info(f"Query: {query}")
        if self.use_retrieved_context:
            context = self.retriever.query(query=query)
            sources = set([info["source"] for info in context])
            context_query = "\n".join(
                f"source: {info['source']}\ncontent: {info['text']}\n"
                for info in context
            )

            prompt = (
                "[INST] You are a scikit-learn expert that should be able to answer"
                " machine-learning question.\n\nAnswer to the query below using the"
                " additional provided content. The additional content is composed of"
                " the HTML link to the source and the extracted contextual"
                " information.\n\nBe succinct.\n\nquery:"
                f" {query}\n\n{context_query} [/INST]."
            )
        else:
            sources = None
            prompt = (
                "[INST] You are a scikit-learn expert that should be able to answer "
                "machine-learning question.\n\n"
                "Answer to the following query.\n\n"
                "Be succinct.\n\n"
                f"query: {query} [/INST]."
            )
        logger.info(f"The final prompt is:\n{prompt}")
        return self.llm(prompt, **prompt_kwargs), sources
