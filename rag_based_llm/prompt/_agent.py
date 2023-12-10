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
        api_semantic_retriever,
        api_lexical_retriever,
    ):
        self.llm = llm
        self.api_semantic_retriever = api_semantic_retriever
        self.api_lexical_retriever = api_lexical_retriever

    def __call__(self, query, **prompt_kwargs):
        # FIXME: We need to come with a design such that it is generic enough to accept
        # different prompt strategies.
        max_tokens = prompt_kwargs.get("max_tokens", 1024)
        prompt = (
            "[INST] Summarize the query provided by extracting keywords from it. "
            f"query: {query}[/INST]"
        )
        response = self.llm(trim(prompt, max_tokens=max_tokens), **prompt_kwargs)
        keywords = response["choices"][0]["text"].strip()
        api_semantic_context = self.api_semantic_retriever.k_neighbors(query)
        api_lexical_context = self.api_lexical_retriever.k_neighbors(keywords)

        # Keep the context that have common sources between the semantic and lexical
        # retrievers.
        api_common_sources = set(
            api["source"] for api in api_semantic_context
        ).intersection(api["source"] for api in api_lexical_context)
        context = "\n".join(
            f"source: {api['source']} \n content: {api['text']}\n"
            for api in api_semantic_context
            if api["source"] in api_common_sources
        )
        prompt = (
            "[INST] Answer to the query related to scikit-learn using the following "
            "pair of content and source. Be succinct. \n"
            f"query: {query}\n"
            f"context: {context} [/INST]."
        )
        response = self.llm(trim(prompt, max_tokens=max_tokens), **prompt_kwargs)
        return (
            "\n".join(wrap(response["choices"][0]["text"].strip(), width=80))
            + "\n\nSource(s):\n"
            + "\n".join(api_common_sources)
        )
