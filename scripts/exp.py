# %% [markdown]
# Not bothering with a installable package for now. For now, important the modules
# directly from the source code.

# %%
import sys
from pathlib import Path

import joblib

sys.path.append(str(Path(__file__).parent.parent))

# %% [markdown]
# Define the training pipeline that extract the text chunks from the API documentation
# and then embed them using a sentence transformer.

# %%
from rag_based_llm.scraping import APIDocExtractor
from rag_based_llm.embedding import SentenceTransformer
from rag_based_llm.retrieval import SemanticRetriever
from sklearn.pipeline import Pipeline

embedding = SentenceTransformer(model_name_or_path="thenlper/gte-large", device="mps")
pipeline = Pipeline(
    steps=[
        ("extractor", APIDocExtractor(chunk_size=700, chunk_overlap=50, n_jobs=-1)),
        ("semantic_retriever", SemanticRetriever(embedding=embedding, n_neighbors=5)),
    ]
)
pipeline

# %% [markdown]
# Fit the pipeline.

# %%
API_DOC = Path(
    "/Users/glemaitre/Documents/packages/scikit-learn/doc/_build/html/stable/"
    "modules/generated"
)
pipeline.fit(API_DOC)

# %% [markdown]
# Save the semantic retriever to be used in the inference time.

# %%
path_api_semantic_retriever = "../models/api_semantic_retrieval.joblib"
joblib.dump(
    pipeline.named_steps["semantic_retriever"],
    path_api_semantic_retriever,
)

# %% [markdown]
# Inference time. Load the vector database. The database will be used to retrieve the
# most pertinent context from the API documentation.

# %%
path_api_semantic_retriever = "../models/api_semantic_retrieval.joblib"
api_semantic_retriever = joblib.load(path_api_semantic_retriever)

# %% [markdown]
# Load the LLM model to be used to generate the response to the query. Instantiate an
# agent that will be used to query the model and retrieve the context from the semantic
# retriever.

# %%
from rag_based_llm.prompt import QueryAgent
from llama_cpp import Llama

model_path = "../models/mistral-7b-instruct-v0.1.Q6_K.gguf"
llm = Llama(
    model_path=model_path,
    device="mps",
    n_gpu_layers=1,
    n_threads=4,
    n_ctx=4096,
)
agent = QueryAgent(
    llm=llm,
    api_semantic_retriever=api_semantic_retriever,
)

# %% [markdown]
# Query the agent with a question.

# %%
query = "What is the possible value for the strategy parameter in the DummyClassifier?"
response = agent(query, max_tokens=4096, temperature=0.1)

# %% [markdown]
# Print the response.

# %%
print(response["choices"][0]["text"])

# %%
