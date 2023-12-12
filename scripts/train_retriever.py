# %% [markdown]
# # Training the retriever
#
# This notebook will train the lexical and semantic retriever and store them.
#
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
from sklearn.pipeline import Pipeline

from rag_based_llm.embedding import SentenceTransformer
from rag_based_llm.retrieval import SemanticRetriever
from rag_based_llm.scraping import APIDocExtractor

embedding = SentenceTransformer(model_name_or_path="thenlper/gte-large", device="mps")
pipeline = Pipeline(
    steps=[
        ("extractor", APIDocExtractor(chunk_size=700, chunk_overlap=50, n_jobs=-1)),
        ("semantic_retriever", SemanticRetriever(embedding=embedding, top_k=15)),
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
joblib.dump(pipeline.named_steps["semantic_retriever"], path_api_semantic_retriever)

# %% [markdown]
# Create a lexical retriever to match some keywords. We take a very long chunk to be
# sure that the keywords are present in the chunk.

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from rag_based_llm.retrieval import BM25Retriever
from rag_based_llm.scraping import APIDocExtractor

count_vectorizer = CountVectorizer(ngram_range=(1, 5))
pipeline = Pipeline(
    steps=[
        ("extractor", APIDocExtractor(chunk_size=1_500, chunk_overlap=200, n_jobs=-1)),
        (
            "lexical_retriever",
            BM25Retriever(count_vectorizer=count_vectorizer, top_k=15),
        ),
    ]
)
API_DOC = Path(
    "/Users/glemaitre/Documents/packages/scikit-learn/doc/_build/html/stable/"
    "modules/generated"
)
pipeline.fit(API_DOC)
