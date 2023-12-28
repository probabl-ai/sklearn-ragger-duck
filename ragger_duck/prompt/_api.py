import logging

from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import HasMethods

logger = logging.getLogger(__name__)


class APIPromptingStrategy(BaseEstimator):
    """Prompting strategy for answering API-related query.

    Once we retrieve the API-related context, we request to answer the query using the
    context.

    Parameters
    ----------
    llm : llm instance
        The language model to use for the prompting. We expect the model to implement
        a `__call__` method that takes a prompt and returns a response. It should be an
        "Instruct"-based model.

    api_retriever : retriever instance
        The API retriever to use for retrieving the API-related context. We expect the
        retriever to implement a `query` method.
    """

    _parameter_constraints = {
        "llm": [HasMethods(["__call__"])],
        "api_retriever": [HasMethods(["query"])],
    }

    def __init__(self, *, llm, api_retriever):
        self.llm = llm
        self.api_retriever = api_retriever

    def __call__(self, query, **prompt_kwargs):
        logger.info(f"Query: {query}")
        logger.info(
            f"Retriever {self.api_retriever.__class__.__name__} does not support"
            " lexical queries"
        )
        context = self.api_retriever.query(query=query)
        sources = set([api["source"] for api in context])
        context_query = "\n".join(
            f"source: {api['source']}\ncontent: {api['text']}\n" for api in context
        )

        prompt = (
            "[INST] You are a scikit-learn expert that should be able to answer "
            "machine-learning question.\n\n"
            "Answer to the query below using the additional provided content. "
            "The additional content is composed of the HTML link to the source and the "
            "extracted contextual information.\n\n"
            "Be succinct.\n\n"
            f"query: {query}\n\n"
            f"{context_query} [/INST]."
        )
        logger.info(f"The final prompt is:\n{prompt}")
        return self.llm(prompt, **prompt_kwargs), sources
