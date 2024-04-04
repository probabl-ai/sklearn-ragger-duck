# Path to the HTML API documentation
API_DOC_PATH = (
    "/Users/glemaitre/Documents/packages/scikit-learn/doc/_build/html/stable/"
    "modules/generated"
)
# Path to the HTML User Guide documentation
USER_GUIDE_DOC_PATH = (
    "/Users/glemaitre/Documents/packages/scikit-learn/doc/_build/html/stable"
)
USER_GUIDE_EXCLUDE_FOLDERS = ["modules/generated", "auto_examples/", "tutorial/"]

# Path to cache the embedding and models
CACHE_PATH = "../models"

# Path to store the retriever once trained
API_SEMANTIC_RETRIEVER_PATH = "../models/api_semantic_retrieval.joblib"
API_LEXICAL_RETRIEVER_PATH = "../models/api_lexical_retrieval.joblib"
USER_GUIDE_SEMANTIC_RETRIEVER_PATH = "../models/user_guide_semantic_retrieval.joblib"
USER_GUIDE_LEXICAL_RETRIEVER_PATH = "../models/user_guide_lexical_retrieval.joblib"
