import pytest

from ragger_duck.prompt import BasicPromptingStrategy


class DummyRetriever:
    def fit(self, X=None, y=None):
        return self

    def query(self, query):
        return [
            {"source": "https://dummy.com", "text": "dummy context"},
        ] * 3


class DummyLLM:
    def __init__(self, n_message=10):
        self.n_message = n_message

    def __call__(self, prompt, **prompt_kwargs):
        self._stored_prompt = prompt
        for _ in range(self.n_message):
            yield "dummy message"


@pytest.mark.parametrize("use_retrieved_context", [True, False])
def test_basic_prompting_strategy(use_retrieved_context):
    """Check the general behaviour of the BasicPromptingStrategy."""

    llm = DummyLLM()
    retriever = DummyRetriever().fit()
    prompter = BasicPromptingStrategy(
        llm=llm, retriever=retriever, use_retrieved_context=use_retrieved_context
    ).fit()

    query = "dummy query"
    responses, source = prompter(query)
    if use_retrieved_context:
        assert source == {
            "https://dummy.com",
        }
    else:
        assert source is None

    for response in responses:
        assert response == "dummy message"

    # await for consuming the generator
    if use_retrieved_context:
        assert "dummy context" in prompter.llm._stored_prompt
    else:
        assert "dummy context" not in prompter.llm._stored_prompt
