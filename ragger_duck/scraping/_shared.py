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
            if elem.strip():
                texts.append(elem.strip())
        elif elem.name == "section":
            continue
        else:
            # Remove the duplicated line breaks on the fly
            text = re.sub(r"\n\s+", "\n", elem.get_text(" "))
            # Remove the duplicated spaces on the fly
            text = re.sub(r" \s+", " ", text)
            texts.append(text.strip())
    return "\n".join(texts)
