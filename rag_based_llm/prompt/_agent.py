class QueryAgent:
    def __init__(
        self,
        *,
        llm,
        retriever,
    ):
        self.llm = llm
        self.retriever = retriever

    def __call__(self, query, **prompt_kwargs):
        # FIXME: We need to come with a design such that it is generic enough to accept
        # different prompt strategies.
        prompt_kwargs.get("max_tokens", 1024)
        prompt = (
            "[INST] Summarize the query provided by extracting keywords from it. "
            "Only list the keywords only separated by a comma. \n"
            f"query: {query}[/INST]"
        )

        # do not create a stream generator
        local_prompt_kwargs = prompt_kwargs.copy()
        local_prompt_kwargs["stream"] = False
        response = self.llm(prompt, **local_prompt_kwargs)
        keywords = response["choices"][0]["text"].strip()

        context = self.retriever.query(
            query=query, lexical_query=keywords, semantic_query=query
        )
        sources = set([api["source"] for api in context])
        context_query = "\n".join(
            f"source: {api['source']} \n content: {api['text']}\n" for api in context
        )
        prompt = (
            "[INST] Answer to the query related to scikit-learn using the following "
            "pair of content and source. The context is provided from the most "
            "relevant to the least relevant. Use this priority to answer to the query. "
            "You have to finish your answer with the source that is an https link of "
            "the content that you used to answer the query. Be succinct. \n"
            f"query: {query}\n"
            f"context: {context_query} [/INST]."
        )
        return self.llm(prompt, **prompt_kwargs), sources
