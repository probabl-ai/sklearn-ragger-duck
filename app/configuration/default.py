# Retriever parameters
API_SEMANTIC_RETRIEVER_PATH = "../models/user_guide_semantic_retrieval.joblib"
API_SEMANTIC_TOP_K = 5
API_LEXICAL_RETRIEVER_PATH = "../models/user_guide_lexical_retrieval.joblib"
API_LEXICAL_TOP_K = 5
CROSS_ENCODER_PATH = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_THRESHOLD = 2.0
CROSS_ENCODER_MIN_TOP_K = 5
CROSS_ENCODER_MAX_TOP_K = 10

# Device parameters
DEVICE = "mps"

# LLM parameters
LLM_PATH = "../models/mistral-7b-instruct-v0.1.Q6_K.gguf"
TEMPERATURE = 0.1
TOP_P = 0.6
TOP_K = 40
REPETATION_PENALTY = 1.176
CONTEXT_TOKENS = 8192
MAX_RESPONSE_TOKENS = 8192
N_THREADS = 6
GPU_LAYERS = 1
