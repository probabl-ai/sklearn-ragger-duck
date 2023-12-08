# %%
import sys

sys.path.append("/Users/glemaitre/Documents/scratch/rag_based_llm")

# %%
from pathlib import Path

API_DOC = Path(
    "/Users/glemaitre/Documents/packages/scikit-learn/doc/_build/html/stable/"
    "modules/generated"
)

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

# %%
pipeline.fit(API_DOC)

# %%
import joblib

joblib.dump(
    pipeline.named_steps["semantic_retriever"],
    "../models/api_semantic_retrieval.joblib",
)

# %%
import joblib

api_semantic_retriever = joblib.load("../models/api_semantic_retrieval.joblib")

# %%
from rag_based_llm.prompt import QueryAgent
from llama_cpp import Llama

llm = Llama(
    model_path="../models/mistral-7b-instruct-v0.1.Q6_K.gguf",
    device="mps",
    n_gpu_layers=1,
    n_threads=4,
    n_ctx=4096,
)
api_semantic_retriever.set_params(n_neighbors=5)
agent = QueryAgent(
    llm=llm,
    api_semantic_retriever=api_semantic_retriever,
)

# %%
query = "What is the possible value for the strategy parameter in the DummyClassifier?"
response = agent(query, max_tokens=4096, temperature=0.1)

# %%
print(response["choices"][0]["text"])

# %%
