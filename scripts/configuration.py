# Path to the HTML API documentation
API_DOC_PATH = "../local_sklearn/scikit-learn/doc/_build/html/stable/modules/generated"
# Path to the HTML User Guide documentation
USER_GUIDE_DOC_PATH = "../local_sklearn/scikit-learn/doc/_build/html/stable"
USER_GUIDE_EXCLUDE_FOLDERS = [
    "_downloads/",
    "_images/",
    "_sources/",
    "_static/",
    "auto_examples/",
    "binder/",
    "developers/index.html",
    "modules/generated",
    "notebooks/",
    "sg_execution_times",
    "testimonials/",
    "tutorial/",
    "_contributors.",
]
# Path to the sphinx-gallery python examples
GALLERY_EXAMPLES_PATH = "../local_sklearn/scikit-learn/examples"

# Path to cache the embedding and models
CACHE_PATH = "../models"

# Path to store the retriever once trained
API_SEMANTIC_RETRIEVER_PATH = "../models/api_semantic_retrieval.joblib"
API_LEXICAL_RETRIEVER_PATH = "../models/api_lexical_retrieval.joblib"
USER_GUIDE_SEMANTIC_RETRIEVER_PATH = "../models/user_guide_semantic_retrieval.joblib"
USER_GUIDE_LEXICAL_RETRIEVER_PATH = "../models/user_guide_lexical_retrieval.joblib"
GALLERY_SEMANTIC_RETRIEVER_PATH = "../models/gallery_semantic_retrieval.joblib"
GALLERY_LEXICAL_RETRIEVER_PATH = "../models/gallery_lexical_retrieval.joblib"

# Parameters for the scraper
CHUNK_SIZE = 1_500
CHUNK_OVERLAP = 10
