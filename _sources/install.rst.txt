.. _getting_started:

###############
Getting Started
###############

Use the Ragger Duck library
===========================

Since this is a Sunday afternoon project, there is not yet a way to install the
package. An easy and dirty way is to add the package into your path for the moment::

  import sys

  path_to_package = "/path/to/ragger_duck"
  sys.path.append(path_to_package)

You can check the file `requirements.txt` for the required packages. We don't yet
provide an easy way to install GPU-enabled packages.

Deploy Ragger Duck Web Console
==============================

To ease the deployment, we rely on `pixi`. Refer to following
`link <https://pixi.sh/#installation>`_ for installing `pixi`.

In the latest stage, `pixi` will be in charge to create the Python environment to
deploy the Web Console.

Them, follow the following steps to deploy the Ragger Duck Web Console.

Build the scikit-learn documentation
------------------------------------

First, we need to have the HTML source file of the API documentation of scikit-learn.
You can clone the scikit-learn repository and build the documentation. Refer to the
following `link <https://scikit-learn.org/dev/developers/contributing.html#building-the-documentation>`_
for building the documentation of scikit-learn:

The API documentation will be located in the folder
`doc/_build/html/stable/modules/generated`.

Train the semantic and lexical retrievers
-----------------------------------------

You need to train the retrievers that will search for the context. You need to modify
the file `scripts/configuration.py` to specify the path to the API documentation.
For the moment, you can let the other variables as-is.

Then, Train the retrievers by running the following command::

  pixi run train-retrievers

from the root folder of the project.

Download the Large Language Model
---------------------------------

You need to get a Large Language Model (LLM). For testing purpose, you can get the
mistral-7b-instruct-v0.1.Q6_K.gguf model that is available at this
`link <https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main>`_.

Launch the Web Console
----------------------

Before to launch the web console, you need to modify the file
`app/configuration/default.py` to provide the links to the retriever models and the
LLM.

Then, Launch the Web Console by running the following command::

  pixi run start-ragger-duck

from the root folder of the project.

Then, you can access the Web Console at the following address::

  http://localhost:8123
