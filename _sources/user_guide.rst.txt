.. title:: User guide: contents

.. _user_guide:

==========
User Guide
==========

Scraping
========

The scraping module provides some simple estimator that extract meaningful
documentation from the documentation website.

:class:`~ragger_duck.scraping.APIDocExtractor` is a scraper that loads the
HTML pages and extract the documentation from the main API section. One can
provide a `chunk_size` and `chunk_overlap` to further split the documentation
sections into smaller chunks.

:class:`~ragger_duck.scraping.APINumPyDocExtractor` is a more advanced scraper
that uses `numpydoc` and it scraper to extract the documentation. Indeed, the
`numpydoc` scraper will parse the different sections and we build meaningful
chunks of documentation from the parsed sections. While, we don't control for
the chunk size, the chunks are build such that they contain information only
of a specific parameter and always refer to the class or function. We hope that
scraping in such way can remove ambiguity that could exist when building chunks
without any control.
