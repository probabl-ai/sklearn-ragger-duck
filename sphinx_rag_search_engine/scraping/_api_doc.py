"""Utilities to scrape the API documentation."""
import re
from numbers import Integral
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._param_validation import StrOptions


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


def extract_api_doc(api_doc_folder, *, return_as="generator", n_jobs=None):
    """Extract text from each HTML API file from a folder

    Parameters
    ----------
    api_doc_folder : :class:`pathlib.Path`
        The path to the API documentation folder.

    return_as : {"generator", "generator_unordered", "list"}, \
        default="generator"
        The type of object to return.

        - If `"generator"`, is returned in an ordered fashion as provided by the input;
        - If `"generator_unordered"`, is returned in an unordered fashion;
        - If `"list"`, is returned as a list.

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
    parallel = Parallel(n_jobs=n_jobs, return_as=return_as)
    return parallel(
        delayed(extract_api_doc_from_single_file)(api_html_file)
        for api_html_file in api_doc_folder.glob("*.html")
    )


class APIDocExtractor(BaseEstimator, TransformerMixin):
    """Extract text from the API documentation.

    This function can process classes and functions.

    Parameters
    ----------
    output_type : {"generator", "generator_unordered", "list"}, \
        default="generator"
        The type of object to return.

        - If `"generator"`, is returned in an ordered fashion as provided by the input;
        - If `"generator_unordered"`, is returned in an unordered fashion;
        - If `"list"`, is returned as a list.

    n_jobs : int, default=None
        The number of jobs to run in parallel. If None, then the number of jobs is set
        to the number of CPU cores.
    """

    _parameter_constraints = {
        "output_type": [StrOptions({"generator", "generator_unordered", "list"})],
        "n_jobs": [Integral, None],
    }

    def __init__(self, *, output_type="generator", n_jobs=None):
        self.output_type = output_type
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
        return extract_api_doc(X, return_as=self.output_type, n_jobs=self.n_jobs)

    def _more_tags(self):
        return {"X_types": ["string"], "stateless": True}
