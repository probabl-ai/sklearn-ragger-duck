import inspect
import logging

from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import HasMethods

logger = logging.getLogger(__name__)


class APIPromptingStrategy(BaseEstimator):
    """Prompting strategy for answering API-related query.

    We use the following prompting strategy:

    - If the retriever support a lexical query, we first extract the keywords from
      the query and use them specifically for the lexical search. Use the full query
      for the semantic search.
    - If the retriever does not support a lexical query, we use the full query as-is.

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
        signature_retriever = inspect.signature(self.api_retriever.query)
        if "lexical_query" in signature_retriever.parameters:
            logger.info(f"Retriever {self.api_retriever} supports lexical queries")
            prompt = (
                "[INST] Summarize the query provided by extracting keywords from it. "
                "Only list the keywords only separated by a comma. \n"
                f"query: {query}[/INST]"
            )

            # do not create a stream generator
            local_prompt_kwargs = prompt_kwargs.copy()
            local_prompt_kwargs["stream"] = False
            logger.info("Prompting to get keywords from the query")
            response = self.llm(prompt, **local_prompt_kwargs)
            keywords = response["choices"][0]["text"].strip()
            logger.info(f"Keywords: {keywords}")
            context = self.api_retriever.query(
                query=query, lexical_query=keywords, semantic_query=None
            )
        else:
            logger.info(
                f"Retriever {self.api_retriever} does not support lexical queries"
            )
            context = self.api_retriever.query(query=query)
        sources = set([api["source"] for api in context])
        context_query = "\n".join(
            f"source: {api['source']}\ncontent: {api['text']}\n" for api in context
        )

        prompt = (
            "[INST] You are a scikit-learn expert that should be able to answer "
            "machine-learning question.\n\n "
            "Answer to the query below using the additional provided content."
            "The additional content is composed of the HTML link to the source and the "
            "extracted text to be used.\n\n"
            "Be succinct.\n\n"
            f"query: {query}\n"
            f"content: {context_query} [/INST]."
        )
        logger.info(f"The final prompt is:\n{prompt}")
        return self.llm(prompt, **prompt_kwargs), sources
