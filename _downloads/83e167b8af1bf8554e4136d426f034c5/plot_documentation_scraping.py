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
# The chunks are stored in a list of dictionaries.
print(f"Chunks is {type(chunks)}")
print(f"A chunk is {type(chunks[0])}")

# %%
# A chunk contains 2 keys: `"source"` that is the HTML source page and `"text"` that is
# the extracted text.
chunks[0].keys()

# %%
# For the API documentation, we use `numpydoc` to generate meaningful chunks. For
# instance, this is the first chunk of text.
print(chunks[0]["text"])

# %%
# The first line of the chunk corresponds to the estimator or class name and its
# module. This information is useful to disambiguate the documentation when using an
# LLM: sometimes we can have multiple parameters name defined in different classes or
# functions. An LLM will tend to summarize the information coming from the different
# chunks. However, if we provide the class or function name and this information is
# present in the user prompt, then the LLM is likely to generate a more accurate
# answer.
#
# Since `numpydoc` offer a structured information based on the sections of the
# docstring, we therefore use these sections and create hand-crafted chunks that we
# find meaningful in regards to the API documentation.
