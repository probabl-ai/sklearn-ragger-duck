.. _getting_started:

###############
Getting Started
###############

Deploy Ragger Duck
==================

To ease the deployment, we rely on `pixi`. Refer to following
`link <https://pixi.sh/#installation>`_ for installing `pixi` but in short, for the
currently supported platform, the following should be enough::

  curl -fsSL https://pixi.sh/install.sh | bash

In the latest stage, `pixi` will be in charge to create the Python environments to
build the scikit-learn documentation, train the retrievers, and launch the Web Console.
We already setup several environments for you depending on the platform and hardware
at your disposal:

- `cpu`: this is a cross-platform environments (i.e. linux and MacOS on x86_64 and
  arm64);
- `mps`: this is an environment for MacOS on M1/M2/M3 chips;
- `cuda-12-1`: this is an environment for linux on x86_64 machine with GPU support.
  We used it to make experiment on Scaleway instance that provides an L4 GPU.
- `cuda-11-7`: similar to `cuda-12-1` but relying on cuda 11.7 instead of 12.1.

Note that you can modify the `pixi.toml` to create your own environments since the
cuda version used in the `cuda-12-1` or `cuda-11-7` environment might not suits your
needs.


Cloning the project
-------------------

The GitHub repository self-contained all the necessary source files for building the
RAG. You need to clone the repository in a recursive way to get the scikit-learn
source files as a submodule::

  git clone --recursive git@github.com:glemaitre/sklearn-ragger-duck.git

Install dependencies using `pixi`
---------------------------------

The subsequent steps will require some dependencies to be installed. They are defined
in the `pixi.lock` file and can be installed using `pixi install`. However, you need
to specify which environment you want to use as stated in the previous section. Here,
we will use the `cpu` environment::

  pixi install --frozen -e cpu

Build the scikit-learn documentation
------------------------------------

First, we need to build the scikit-learn documentation since some of the retrievers
will rely on the HTML generated pages. You can build the documentation by running the
following command::

  pixi run --frozen build-doc-sklearn

Train the semantic and lexical retrievers
-----------------------------------------

We need to train a set of lexical and semantic retrievers on the API documentation,
the user guide, and the gallery of examples. We will have different retrievers
for each of these type of documentation. You can refer :ref:`user_guide` for more
details on the strategy used to train the retrievers.

You can launch the training of the retrievers by running the following command::

  pixi run --frozen train-retrievers

Pixi might propose you to select a specific environment to make the training. You can
also specify the environment by running the following command::

  pixi run --frozen -e cpu train-retrievers

Download the Large Language Model
---------------------------------

You need to get a Large Language Model (LLM). For testing purpose, you can get the
Mistral 7b model by running the following command::

  pixi run --frozen fetch-mistral-7b

Launch the Web Console
----------------------

Now, you are all set to start the web console.

Then, Launch the Web Console by running the following command::

  pixi run --frozen start-ragger-duck

You will also be required to select an environment depending on which hardware you want
to offload the LLM.

Then, you can access the Web Console at the following address::

  http://127.0.0.1:8123

Use the Ragger Duck library
===========================

When using `pixi` as discussed earlier, Ragger Duck is installed in editable mode in the
environment. However, we also make Ragger Duck installable via `pip`::

  pip install -e .

However, we don't install any of the dependencies since it is hardware dependent and
can be better handled with `pixi`.
