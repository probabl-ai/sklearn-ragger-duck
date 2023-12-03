"""Utilities to scrape the API documentation."""
from bs4 import BeautifulSoup, NavigableString


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
            texts.append(elem.get_text().strip())
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


def extract_api_doc(api_doc_folder):
    """Extract text from each HTML API file from a folder

    Parameters
    ----------
    api_doc_folder : :class:`pathlib.Path`
        The path to the API documentation folder.
    """
