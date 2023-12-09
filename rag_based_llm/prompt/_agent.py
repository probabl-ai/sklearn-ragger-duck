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

    _system_content = (
        "Answer the query with your knowledge and the context provided. Use only the "
        "most concordant context. At the end of your answer, please add the source of "
        "your answer. It should be a link to the website the most repeated."
    )

    def __init__(
        self,
        *,
        llm,
        api_semantic_retriever=None,
    ):
        self.llm = llm
        self.api_semantic_retriever = api_semantic_retriever

    @property
    def system_content(self):
        """System content to be used in the prompt."""
        return self._system_content

    @system_content.setter
    def system_content(self, value):
        self._system_content = value

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
        prompt_template = (
            "[INST] {system} \n query: {query} \n context: {context} \n "
            "source: {source} [/INST]"
        )
        if self.api_semantic_retriever is not None:
            system_content = self.system_content
            api_context = self.api_semantic_retriever.k_neighbors(query)[0]
            api_content = "\n".join(api["text"] for api in api_context)
            api_source = "\n".join(api["source"] for api in api_context)
        else:
            system_content, api_content = "", ""

        max_tokens = prompt_kwargs.get("max_tokens", 1024)
        prompt = prompt_template.format(
            system=system_content,
            query=query,
            context=api_content,
            source=api_source,
        )
        print(prompt)
        return self.llm(trim(prompt, max_tokens=max_tokens), **prompt_kwargs)
