"""Utilities to scrape the gallery of examples."""

import logging
import re
from itertools import chain
from numbers import Integral

from joblib import Parallel, delayed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils._param_validation import Interval
from sphinx_gallery.py_source_parser import split_code_and_text_blocks

from ._shared import _chunk_document

SKLEARN_EXAMPLES_URL = "https://scikit-learn.org/stable/auto_examples/"
logger = logging.getLogger(__name__)


def _example_gallery_path_to_example_gallery_url(path):
    """Convert an example path to an example URL.

    Parameters
    ----------
    path : :class:`pathlib.Path`
        The path to the Example documentation.

    Returns
    -------
    str
        The Examples URL.
    """
    # Find the examples folder to reconstruct the URL from this point
    for parent in path.parents:
        if parent.name == "examples":
            break
    # We scrap the .py file and return the URL to the HTML file
    return SKLEARN_EXAMPLES_URL + str(path.relative_to(parent).with_suffix(".html"))


def _split_block_if_contains_section(block):
    """Split a block of text into several blocks by detection rst section."""
    # We only split at the first level that is represented by dashes
    section_expression = "(?m)\n-+\n"
    matches = re.finditer(section_expression, block)
    start_indices = []
    end_indices = []
    for m in matches:
        start_idx = block[: m.start()].rfind("\n", 1)
        start_indices.append(0 if start_idx == -1 else start_idx)
        if len(start_indices) <= 1:
            continue
        end_indices.append(start_idx - 1)
    end_indices.append(len(block))
    return [block[start:end] for start, end in zip(start_indices, end_indices)]


def _split_blocks_at_sections(file_path):
    """Given an example file, split the text blocks at the sections."""
    blocks = []
    for block in split_code_and_text_blocks(file_path, return_node=False)[1]:
        if block[0] == "text":
            if blocks_section := _split_block_if_contains_section(block[1]):
                for bs in blocks_section:
                    blocks.append(("text", bs))
            else:
                blocks.append(block)
        else:
            blocks.append(block)
    return blocks


def _merge_blocks_per_section(blocks):
    """Merge the text and code blocks together within each section."""
    bounds_section_block = [0]
    for block_id, block in enumerate(blocks):
        if block[0] == "text":
            if re.search("\n-+\n", block[1]) is not None:
                bounds_section_block.append(block_id)
    bounds_section_block.append(len(blocks))

    merged_chunks = []
    for start_block_id, end_block_id in zip(
        bounds_section_block[:-1], bounds_section_block[1:]
    ):
        blocks_to_merge = blocks[start_block_id:end_block_id]
        merged_chunks.append("".join(block[1] for block in blocks_to_merge))
    return merged_chunks


def _extract_single_example(file_path):
    """Extract chunks of text and sources from a single example file.

    Two strategies are implemented to extract the text from the examples:

    - When an example contains only two blocks (a summary and the code), we return
      a list containing the two blocks;
    - When an example contains multiple blocks of text and code, we first detect the
      sections in the text and then merge the text and code blocks together within
      each section since the narration is related.

    Parameters
    ----------
    file_path : :class:`pathlib.Path`
        The path to the example file.

    Returns
    -------
    output : list
        A list of dictionaries containing the source and text of the example.
    """
    file_splitted = split_code_and_text_blocks(file_path)[1]
    n_blocks = len(file_splitted)
    if n_blocks > 2:
        # The example is a multi-block example and it makes sense to merge together
        # parts belonging to the same section. Later, if a section is too long, then
        # it will be chunked outside of this function.
        blocks = _split_blocks_at_sections(file_path)
        return [
            {
                "source": _example_gallery_path_to_example_gallery_url(file_path),
                "text": block_content,
            }
            for block_content in _merge_blocks_per_section(blocks)
        ]
    else:
        # The example is only a code block containing a summary and some code.
        # No need for an extra merge of the text blocks.
        return [
            {
                "source": _example_gallery_path_to_example_gallery_url(file_path),
                "text": block_content,
            }
            for _, block_content, _ in file_splitted
        ]


class GalleryExampleExtractor(BaseEstimator, TransformerMixin):
    """Extract text from the examples of the gallery.

    Parameters
    ----------
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
        "chunk_size": [Interval(Integral, left=1, right=None, closed="left"), None],
        "chunk_overlap": [Interval(Integral, left=0, right=None, closed="left")],
        "n_jobs": [Integral, None],
    }

    def __init__(self, *, chunk_size=300, chunk_overlap=50, n_jobs=None):
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
            output = list(
                chain.from_iterable(
                    _extract_single_example(example)
                    for example in X.rglob("*.py")
                    if not example.name.startswith("_")
                )
            )
        else:
            output = list(
                chain.from_iterable(
                    Parallel(n_jobs=self.n_jobs, return_as="generator")(
                        delayed(_chunk_document)(self.text_splitter_, document)
                        for example_file in X.rglob("*.py")
                        if not example_file.name.startswith("_")
                        for document in _extract_single_example(example_file)
                        if document is not None
                    )
                )
            )
        if not output:
            raise ValueError(
                "No documentation from the examples was extracted. Please check the "
                "input folder."
            )
        return output

    def _more_tags(self):
        return {"X_types": ["string"], "stateless": True}
