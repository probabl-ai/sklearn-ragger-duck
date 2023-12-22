"""SentenceTransformer with a scikit-learn API."""
import logging
import time

from sentence_transformers import SentenceTransformer as SentenceTransformerBase
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class SentenceTransformer(BaseEstimator, TransformerMixin):
    """Sentence transformer that embeds sentences to embeddings.

    This is a thin wrapper around :class:`sentence_transformers.SentenceTransformer`
    that follows the scikit-learn API and thus can be used inside a scikit-learn
    pipeline.

    Parameters
    ----------
     model_name_or_path : str, default=None
        If it is a filepath on disc, it loads the model from that path. If it is not a
        path, it first tries to download a pre-trained SentenceTransformer model. If
        that fails, tries to construct a model from Huggingface models repository with
        that name.

    modules : Iterable of nn.Module, default=None
        This parameter can be used to create custom SentenceTransformer models from
        scratch.

    device : str, default=None
        Device (e.g. "cpu", "cuda", "mps") that should be used for computation. If None,
        checks if a GPU can be used.

    cache_folder : str, default=None
        Path to store models.

    use_auth_token : bool or str, default=None
        HuggingFace authentication token to download private models.

    batch_size : int, default=32
        The batch size to use during `transform`.

    show_progress_bar : bool, default=True
        Whether to show a progress bar or not during `transform`.
    """

    _parameter_constraints = {
        "model_name_or_path": [str, None],
        "modules": "no_validation",
        "device": [str, None],
        "cache_folder": [str, None],
        "use_auth_token": [str, bool, None],
        "show_progress_bar": [bool],
    }

    def __init__(
        self,
        model_name_or_path=None,
        modules=None,
        device=None,
        cache_folder=None,
        use_auth_token=None,
        batch_size=32,
        show_progress_bar=True,
    ):
        self.model_name_or_path = model_name_or_path
        self.modules = modules
        self.device = device
        self.cache_folder = cache_folder
        self.use_auth_token = use_auth_token
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

    def fit(self, X=None, y=None):
        """No-op operation, only validate parameters.

        Parameters
        ----------
        X : None
            This parameter is ignored.

        y : None
            This parameter is ignored.

        Returns
        -------
        self
            The fitted estimator.
        """
        self._validate_params()
        self._embedding = SentenceTransformerBase(
            model_name_or_path=self.model_name_or_path,
            modules=self.modules,
            device=self.device,
            cache_folder=self.cache_folder,
            use_auth_token=self.use_auth_token,
        )
        return self

    def transform(self, X):
        """Embed sentences to vectors.

        Parameters
        ----------
        X : str or Iterable of str or dict or length (n_sentences,)
            The sentences to embed.

            - If `str`, a single sentence to embed;
            - If `list` of `str`, a list of sentences to embed;
            - If `list` of `dict`, a list of dictionaries with a key "text" that
              contains the sentence to embed.

        Returns
        -------
        embedding : ndarray of shape (n_sentences, embedding_size)
            The embedding of the sentences.
        """
        if isinstance(X, str):
            X = [X]
        elif isinstance(X[0], dict):
            X = [chunk["text"] for chunk in X]
        start = time.time()
        embedding = self._embedding.encode(
            X,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            # L2-normalize to use dot-product as similarity measure
            normalize_embeddings=True,
        )
        logger.info(f"Embedding done in {time.time() - start:.2f}s")
        return embedding
