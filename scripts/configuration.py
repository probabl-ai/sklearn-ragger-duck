# Device on which the semantic retriever is trained
DEVICE = "mps"  # {"cpu", "cuda", "mps"}

# Path to the HTML API documentation
API_DOC_PATH = (
    "/Users/glemaitre/Documents/packages/scikit-learn/doc/_build/html/stable/"
    "modules/generated"
)

# Path to cache the embedding and models
CACHE_PATH = "../models"

# Path to store the retriever once trained
SEMANTIC_RETRIEVER_PATH = "../models/api_semantic_retrieval.joblib"
LEXICAL_RETRIEVER_PATH = "../models/api_lexical_retrieval.joblib"
