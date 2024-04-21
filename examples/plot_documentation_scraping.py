"""
=================================
Documentation scraping strategies
=================================

This example illustrates how the different documentation scraping strategies work
in `ragger_duck`.
"""

# %%
# API documentation scraping
# --------------------------
# First, we look at the :class:`~ragger_duck.scraping.APINumPyDocExtractor` class. This
# class is used to scrape the API documentation of scikit-learn. It leverages the
# `numpydoc` scraper and create semi-structured chunk of text.
#
# Let's show an example where we scrape the documentation of
# :class:`~sklearn.ensemble.RandomForestClassifier`. Our scrapper requires the HTML
# generated file to infer if this is part of the public API. To do so,s we copied the
# HTML generated file in the folder `toy_documentation/api`. We can therefore process
# this folder.
from pathlib import Path

from ragger_duck.scraping import APINumPyDocExtractor

path_api_doc = Path(".") / "toy_documentation" / "api"
chunks = APINumPyDocExtractor().fit_transform(path_api_doc)

# %%
for chunk in chunks:
    print(f"The source of the chunk is {chunk['source']}\n")
    print(f"{chunk['text']}\n")

# %%
print("hello world")
