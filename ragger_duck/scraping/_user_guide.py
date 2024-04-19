"""Utilities to scrape User Guide documentation."""

import logging
import re
from itertools import chain
from numbers import Integral
from pathlib import Path

from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import Interval

from ._shared import _chunk_document

SKLEARN_USER_GUIDE_URL = "https://scikit-learn.org/stable/"
logger = logging.getLogger(__name__)


def _user_guide_path_to_user_guide_url(path):
    """Convert a User Guide path to an User Guide URL.

    Parameters
    ----------
    path : :class:`pathlib.Path`
        The path to the User Guide documentation.

    Returns
    -------
    str
        The User Guide URL.
    """
    # Find the stable folder to reconstruct the URL from this point
    for parent in path.parents:
        if parent.name == "stable":
            break
    return SKLEARN_USER_GUIDE_URL + str(path.relative_to(parent))


def extract_user_guide_doc_from_single_file(html_file):
    """Extract the text from the User Guide documentation.

    This function can process classes and functions.

    Parameters
    ----------
    html_file : :class:`pathlib.Path`
        The path to the HTML User Guide documentation.

    Returns
    -------
    dict
        A dictionary containing the source and text of the User Guide documentation.
        The dictionary is an empty dictionary if no section was found in the HTML file
        meaning that it does not follow the template from a scikit-learn user guide
        page.
    """
    if not isinstance(html_file, Path):
        raise ValueError(
            "The User Guide HTML file should be a pathlib.Path object. "
            f"Got {html_file!r}."
        )
    if html_file.suffix != ".html":
        raise ValueError(
            f"The file {html_file} is not an HTML file. Please provide an HTML file."
        )
    with open(html_file, "r") as file:
        soup = BeautifulSoup(file, "html.parser")

    text = soup.find("section")
    if text is not None:
        text = text.get_text("")
    else:
        # Case we don't find a section tag. Let's not parse the file it should be
        # something else than a User Guide page.
        return {}  # pragma: no cover
    # Remove line breaks within a paragraph
    newline = re.compile(r"\n\s*")
    text = newline.sub(r"\n", text)
    # Remove the duplicated spaces on the fly
    multiple_spaces = re.compile(" +")
    text = multiple_spaces.sub(" ", text)

    return {
        "source": _user_guide_path_to_user_guide_url(html_file),
        "text": text,
    }


def _extract_user_guide_doc(user_guide_doc_folder, black_listed_folders):
    """Extract text from each HTML User Guide files from a folder

    Parameters
    ----------
    user_guide_doc_folder : :class:`pathlib.Path`
        The path to the User Guide documentation folder.

    black_listed_folders : None or list of str
        A list of folders to exclude from the HTML pages to process.

    Returns
    -------
    list of dict
        A list of dictionaries containing the source and text of the API
        documentation.
    """
    if not isinstance(user_guide_doc_folder, Path):
        raise ValueError(
            "The User Guide documentation folder should be a pathlib.Path object. Got "
            f"{user_guide_doc_folder!r}."
        )

    result = []
    for html_file in user_guide_doc_folder.rglob("*.html"):
        if black_listed_folders is not None:
            if any(folder in str(html_file) for folder in black_listed_folders):
                continue
        extracted_info = extract_user_guide_doc_from_single_file(html_file)
        if extracted_info:
            # empty dictionary if the extraction failed to find a section tag
            result.append(extracted_info)

    return result


class UserGuideDocExtractor(BaseEstimator, TransformerMixin):
    """Extract text from the User Guide documentation.

    This function can process classes and functions.

    Parameters
    ----------
    folders_to_exclude : list of str, default=None
        A list of strings corresponding to folders name to exclude from the HTML pages
        to process.

    chunk_size : int or None, default=300
        The size of the chunks to split the text into. If None, the text is not chunked.

    chunk_overlap : int, default=50
        The overlap between two consecutive chunks.

    n_jobs : int, default=None
        The number of jobs to run in parallel. If None, then the number of jobs is set
        to the number of CPU cores.

    Attributes
    ----------
    text_splitter_ : :class:`langchain.text_splitter.RecursiveCharacterTextSplitter`
        The text splitter to use to chunk the document. If `chunk_size` is None, this
        attribute is None.
    """

    _parameter_constraints = {
        "folders_to_exclude": [None, list],
        "chunk_size": [Interval(Integral, left=1, right=None, closed="left"), None],
        "chunk_overlap": [Interval(Integral, left=0, right=None, closed="left")],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self, *, folders_to_exclude=None, chunk_size=300, chunk_overlap=50, n_jobs=None
    ):
        self.folders_to_exclude = folders_to_exclude
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.n_jobs = n_jobs

    @_fit_context(prefer_skip_nested_validation=False)
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
        if self.chunk_size is not None:
            self.text_splitter_ = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " "],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
            )
        else:
            self.text_splitter_ = None
        return self

    def transform(self, X):
        """Extract text from the API documentation.

        Parameters
        ----------
        X : :class:`pathlib.Path`
            The path to the API documentation folder.

        Returns
        -------
        output : list
            A list of dictionaries containing the source and text of the User Guide
            documentation.
        """
        if self.chunk_size is None:
            output = _extract_user_guide_doc(X, self.folders_to_exclude)
        else:
            output = list(
                chain.from_iterable(
                    Parallel(n_jobs=self.n_jobs, return_as="generator")(
                        delayed(_chunk_document)(self.text_splitter_, document)
                        for document in _extract_user_guide_doc(
                            X, self.folders_to_exclude
                        )
                    )
                )
            )
        if not output:
            raise ValueError(
                "No User Guide documentation was extracted. Please check the "
                "input folder."
            )
        return output

    def _more_tags(self):
        return {"X_types": ["string"], "stateless": True}
