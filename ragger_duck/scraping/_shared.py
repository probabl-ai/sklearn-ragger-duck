import re

from bs4 import NavigableString


def _extract_text_from_section(section):
    """Extract the text from an HTML section.

    Parameters
    ----------
    section : :class:`bs4.element.Tag`
        The HTML section from which to extract the text.

    Returns
    -------
    str or None
        The text extracted from the section. Return None if the section is a
        :class:`bs4.NavigableString`.

    Notes
    -----
    This function was copied from:
    https://github.com/ray-project/llm-applications/blob/main/rag/data.py
    (under CC BY 4.0 license)
    """
    if isinstance(section, NavigableString):
        return None
    texts = []
    for elem in section.children:
        if isinstance(elem, NavigableString):
            text = elem.strip()
        else:
            text = elem.get_text(" ")
        # Remove line breaks within a paragraph
        newline = re.compile(r"\n+")
        text = newline.sub(" ", text)
        # Remove the duplicated spaces on the fly
        multiple_spaces = re.compile(r"\s+")
        text = multiple_spaces.sub(" ", text)
        texts.append(text)
    return " ".join(texts).replace("¶", "\n")


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
