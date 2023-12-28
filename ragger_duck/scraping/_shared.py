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
