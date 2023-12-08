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
from sphinx_rag_search_engine.scraping import APIDocExtractor
from sphinx_rag_search_engine.embedding import SentenceTransformer
from sphinx_rag_search_engine.retrieval import SemanticRetriever
from sklearn.pipeline import Pipeline

embedding = SentenceTransformer(model_name_or_path="thenlper/gte-large", device="mps")
pipeline = Pipeline(
    steps=[
        ("extractor", APIDocExtractor(n_jobs=-1)),
        ("retriever", SemanticRetriever(embedding=embedding, n_neighbors=5)),
    ]
)

# %%
pipeline.fit(API_DOC)

# %%
retriever = pipeline["retriever"]

# %%
retriever.k_neighbors(
    "What is the default value of n_neighbors in KNeighborsClassifier?",
    n_neighbors=5,
)

# %%
import joblib

joblib.dump(retriever, "retriever.joblib")

# %%
