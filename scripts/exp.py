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
from sphinx_rag_search_engine.scraping import extract_api_doc

input_texts = extract_api_doc(API_DOC, n_jobs=-1)
text = [text["text"] for text in input_texts]

# %%
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("thenlper/gte-large")
model.encode(text, show_progress_bar=True)

# %%
