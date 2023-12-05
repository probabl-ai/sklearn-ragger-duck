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
from sklearn.pipeline import Pipeline

pipeline = Pipeline(
    steps=[
        ("extractor", APIDocExtractor(n_jobs=-1)),
        (
            "embedder",
            SentenceTransformer(model_name_or_path="thenlper/gte-large", device="mps"),
        ),
    ]
)

# %%
database = pipeline.fit_transform(API_DOC)

# %%
xxx = SentenceTransformer(model_name_or_path="sentence-transformers/paraphrase-albert-small-v2")
xxx.fit_transform([{"source": "hello world", "text": "hello world"}])

# %%
