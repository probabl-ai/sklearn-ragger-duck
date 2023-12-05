"""Utilities to scrape the API documentation."""
import re
from itertools import chain
from numbers import Integral
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString
from joblib import Parallel, delayed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._param_validation import Interval


SKLEARN_API_URL = "https://scikit-learn.org/stable/modules/generated/"


def _extract_text_from_section(section):
    """Extract the text from an HTML section.

    Parameters
    ----------
    section : :class:`bs4.element.Tag`
        The HTML section from which to extract the text.

    Returns
    -------
    str
        The text extracted from the section.

    Notes
    -----
    This function was copied from:
    https://github.com/ray-project/llm-applications/blob/main/rag/data.py
    (under CC BY 4.0 license)
    """
    texts = []
    for elem in section.children:
        if isinstance(elem, NavigableString):
            if elem.strip():
                texts.append(elem.strip())
        elif elem.name == "section":
            continue
        else:
            # Remove the duplicated line breaks on the fly
            texts.append(re.sub(r"\n\s+", "\n", elem.get_text(" ")))
    return "\n".join(texts)


def _api_path_to_api_url(path):
    """Convert a API path to an API URL.

    Parameters
    ----------
    path : :class:`pathlib.Path`
        The path to the API documentation.

    Returns
    -------
    str
        The API URL.
    """
    return SKLEARN_API_URL + path.name


def extract_api_doc_from_single_file(api_html_file):
    """Extract the text from the API documentation.

    This function can process classes and functions.

    Parameters
    ----------
    api_html_file : :class:`pathlib.Path`
        The path to the HTML API documentation.

    Returns
    -------
    str
        The text extracted from the API documentation.
    """
    if not isinstance(api_html_file, Path):
        raise ValueError(
            f"The API HTML file should be a pathlib.Path object. Got {api_html_file!r}."
        )
    if api_html_file.suffix != ".html":
        raise ValueError(
            f"The file {api_html_file} is not an HTML file. Please provide an HTML "
            "file."
        )
    with open(api_html_file, "r") as file:
        soup = BeautifulSoup(file, "html.parser")
    api_section = soup.section
    return {
        "source": _api_path_to_api_url(api_html_file),
        "text": _extract_text_from_section(api_section),
    }


def extract_api_doc(api_doc_folder, *, n_jobs=None):
    """Extract text from each HTML API file from a folder

    Parameters
    ----------
    api_doc_folder : :class:`pathlib.Path`
        The path to the API documentation folder.

    n_jobs : int, default=None
        The number of jobs to run in parallel. If None, then the number of jobs is set
        to the number of CPU cores.

    Returns
    -------
    generator or list
        A generator or list of dictionaries containing the source and text of the API
        documentation.
    """
    if not isinstance(api_doc_folder, Path):
        raise ValueError(
            f"The API documentation folder should be a pathlib.Path object. Got "
            f"{api_doc_folder!r}."
        )
    return Parallel(n_jobs=n_jobs)(
        delayed(extract_api_doc_from_single_file)(api_html_file)
        for api_html_file in api_doc_folder.glob("*.html")
    )


def _chunk_document(text_splitter, document):
    """Chunk a document into smaller pieces.

    Parameters
    ----------
    text_splitter : :class:`langchain.text_splitter.RecursiveCharacterTextSplitter`
        The text splitter to use to chunk the document.

    document : dict
        A dictionary containing two keys: `text` and `source`. The value associated
        to the `text` key is the text to chunk. The source is propagated to the
        chunks.

    Returns
    -------
    list of dict
        List of dictionary containing the `document` chunked into smaller pieces.
    """
    chunks = text_splitter.create_documents(
        texts=[document["text"]],
        metadatas=[{"source": document["source"]}],
    )
    return [
        {"text": chunk.page_content, "source": chunk.metadata["source"]}
        for chunk in chunks
    ]


class APIDocExtractor(BaseEstimator, TransformerMixin):
    """Extract text from the API documentation.

    This function can process classes and functions.

    Parameters
    ----------
    chunk_size : int, default=300
        The size of the chunks to split the text into.

    chunk_overlap : int, default=50
        The overlap between two consecutive chunks.

    n_jobs : int, default=None
        The number of jobs to run in parallel. If None, then the number of jobs is set
        to the number of CPU cores.
    """

    _parameter_constraints = {
        "chunk_size": [Interval(Integral, left=1, right=None, closed="left")],
        "chunk_overlap": [Interval(Integral, left=0, right=None, closed="left")],
        "n_jobs": [Integral, None],
    }

    def __init__(self, *, chunk_size=300, chunk_overlap=50, n_jobs=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.n_jobs = n_jobs

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
        self.text_splitter_ = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        return self

    def transform(self, X):
        """Extract text from the API documentation.

        Parameters
        ----------
        X : :class:`pathlib.Path`
            The path to the API documentation folder.

        Returns
        -------
        generator
            A generator of dictionaries containing the source and text of the API
            documentation.
        """
        chunked_content = chain.from_iterable(
            Parallel(n_jobs=self.n_jobs, return_as="generator_unordered")(
                delayed(_chunk_document)(self.text_splitter_, document)
                for document in  extract_api_doc(X, n_jobs=self.n_jobs)
            )
        )
        return list(chunked_content)

    def _more_tags(self):
        return {"X_types": ["string"], "stateless": True}
