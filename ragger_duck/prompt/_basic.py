import logging

from sklearn.base import BaseEstimator, _fit_context
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

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X=None, y=None):
        """No-op operation, only validate parameters.

        Parameters
        ----------
        X : None
            This parameter is ignored.

        y : None
            This parameter is ignored.

        Returns
        -------
        self
            The fitted estimator.
        """
        return self

    def __call__(self, query, **prompt_kwargs):
        """Query the LLM model to answer the query.

        Parameters
        ----------
        query : str
            The query to answer.

        prompt_kwargs : dict
            Additional keyword arguments to pass to the LLM model. It is expected that
            `llm` accepts these arguments when calling `llm(prompt, **prompt_kwargs)`.

        Returns
        -------
        stream : generator
            In most cases, we expect `llm` to return a generator such that it can
            stream to the application.

        sources : set of str
            The sources of the retrieved context. If `use_retrieved_context` is False,
            this value is None.
        """
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
                " information.\n\nBe succinct.\n\n"
                "Make sure to use backticks whenever you refer to class, function, "
                "method, or name that contains underscores.\n\n"
                f"query: {query}\n\n{context_query} [/INST]."
            )
        else:
            sources = None
            prompt = (
                "[INST] You are a scikit-learn expert that should be able to answer "
                "machine-learning question.\n\n"
                "Answer to the following query. Be succinct.\n\n"
                "Make sure to use backticks whenever you refer to class, function, "
                "method, or name that contains underscores.\n\n"
                f"query: {query} [/INST]."
            )
        logger.info(f"The final prompt is:\n{prompt}")
        return self.llm(prompt, **prompt_kwargs), sources
