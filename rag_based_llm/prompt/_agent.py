from textwrap import wrap

import tiktoken


def trim(text, max_tokens):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(enc.encode(text)[:max_tokens])


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
        max_tokens = prompt_kwargs.get("max_tokens", 1024)
        prompt = (
            "[INST] Summarize the query provided by extracting keywords from it. "
            "Only list the keywords only separated by a comma. \n"
            f"query: {query}[/INST]"
        )
        response = self.llm(trim(prompt, max_tokens=max_tokens), **prompt_kwargs)
        keywords = response["choices"][0]["text"].strip()

        context = self.retriever.query(
            query=query, lexical_query=keywords, semantic_query=query
        )
        sources = set([api["source"] for api in context])
        context_query = "\n".join(
            f"source: {api['source']} \n content: {api['text']}\n"
            for api in context
        )
        prompt = (
            "[INST] Answer to the query related to scikit-learn using the following "
            "pair of content and source. Be succinct. \n"
            f"query: {query}\n"
            f"context: {context_query} [/INST]."
        )
        response = self.llm(trim(prompt, max_tokens=max_tokens), **prompt_kwargs)
        return (
            response["choices"][0]["text"].strip()
            + "\n\nSource(s):\n"
            + "\n".join(sources)
        )
