.. _text_scraping:

=============
Text Scraping
=============

The scraping module provides some simple estimator that extract meaningful
documentation from the documentation website.

API documentation
=================

:class:`~ragger_duck.scraping.APINumPyDocExtractor` is a more advanced scraper
that uses `numpydoc` and it scraper to extract the documentation. Indeed, the
`numpydoc` scraper will parse the different sections and we build meaningful
chunks of documentation from the parsed sections. While, we don't control for
the chunk size, the chunks are build such that they contain information only
of a specific parameter and always refer to the class or function. We hope that
scraping in such way can remove ambiguity that could exist when building chunks
without any control.

User Guide documentation
========================

:class:`~ragger_duck.scraping.UserGuideDocExtractor` is a scraper that extract
documentation from the user guide. It is a simple scraper that extract
text information from the webpage. Additionally, this text can be chunked.
