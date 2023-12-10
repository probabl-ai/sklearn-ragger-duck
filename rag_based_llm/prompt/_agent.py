import re

import tiktoken


def trim(text, max_tokens):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(enc.encode(text)[:max_tokens])


class QueryAgent:
    """Agent to query the LLM model.

    Parameters
    ----------
    llm : :class:`llama_cpp.Llama`
        The LLM model to be queried.

    api_semantic_retriever : \
            :class:`rag_based_llm.retrieval.SemanticRetriever`
        The semantic retriever to be used to retrieve the most relevant context
        extracted from the API documentation.
    """

    def __init__(
        self,
        *,
        llm,
        api_semantic_retriever,
    ):
        self.llm = llm
        self.api_semantic_retriever = api_semantic_retriever

    def __call__(self, query, **prompt_kwargs):
        """Query the LLM model.

        Parameters
        ----------
        query : str
            The query to be asked to the LLM model.

        prompt_kwargs : dict, default=None
            Additional keyword arguments to be passed to the LLM model. Check the
            `__call__` function of the :class:`llama_cpp.Llama` class to know the valid
            keyword arguments.
        """
        max_tokens = prompt_kwargs.get("max_tokens", 1024)

        api_context = self.api_semantic_retriever.k_neighbors(query)[0]
        context = "\n".join(
            f"source: {api['source']} \n content: {api['text']}\n"
            for api in api_context
        )
        prompt = (
            "[INST] Given the following list of pair of context and source, "
            "select a single source in which the important keywords in the query "
            "also appear in the content linked to the source.\n"
            f"query: {query} \n Pair content/source: \n {context} \n"
            "Only provide the https link.[/INST]"
        )
        print(prompt)
        response = self.llm(trim(prompt, max_tokens=max_tokens), **prompt_kwargs)
        source = response["choices"][0]["text"].strip()
        print(source)
        pattern = r'https.*\.html'
        source = re.search(pattern, source).group()
        prompt = (
            f"[INST] Answer to the query using the following pair of content and "
            f"source. Use principally the content from the following source: {source} "
            "Be succinct."
            f"query: {query}\n"
            f"context: {context}[/INST]."
        )
        response = self.llm(trim(prompt, max_tokens=max_tokens), **prompt_kwargs)
        return response["choices"][0]["text"]
